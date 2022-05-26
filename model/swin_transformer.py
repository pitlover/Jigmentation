from typing import Optional, Union, Tuple
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath

_SWIN_PADDING_MODE = "replicate"


class SwinMLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinWindowing(nn.Module):
    """Module-like wrapper of windowing function."""

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.H: Optional[int] = None  # placeholder
        self.W: Optional[int] = None  # placeholder

    def forward(self):
        raise NotImplementedError

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        window_size = self.window_size
        b, h, w, c = x.shape
        self.H, self.W = h, w

        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        # (b, h, w, c) -> (b, h/r, r, w/r, r, c) -> (b, h/r, w/r, r, r, c) -> (b * h/r * w/r, r, r, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
        return windows

    def window_reverse(self, windows: torch.Tensor,
                       orig_h: Optional[int] = None, orig_w: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            orig_h (int): Height of image
            orig_w (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        orig_h = orig_h if (orig_h is not None) else self.H
        orig_w = orig_w if (orig_w is not None) else self.W
        window_size = self.window_size
        split_h = orig_h // window_size
        split_w = orig_w // window_size
        b = windows.shape[0] // (split_h * split_w)  # actual batch size
        x = windows.view(b, split_h, split_w, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, orig_h, orig_w, -1)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop_prob (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_prob (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim: int,
                 window_size: Union[int, Tuple[int, int]],
                 num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"Dim {dim} is not divisible by num_heads {num_heads}.")
        head_dim = dim // num_heads
        self.attn_scale = math.sqrt(1 / head_dim)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, wh, ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, wh x ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, wh x ww, wh x ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (wh x ww, wh x ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (wh x ww, wh x ww)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj_drop = nn.Dropout(drop_prob)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, nh, n, hd), n == wh x ww

        q *= self.attn_scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # (wh x ww,wh x ww, nh)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nh, wh x ww, wh x ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)  # (b, nh, n, n) x (b, nh, n, nd) = (b, nh, n, nd)
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float, optional): Dropout rate. Default: 0.0
        attn_drop_prob (float, optional): Attention dropout rate. Default: 0.0
        path_drop_prob (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 path_drop_prob: float = 0.0,
                 act_layer=nn.GELU) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if not (0 <= self.shift_size < self.window_size):
            raise ValueError(f"shift_size {shift_size} must in [0, window_size {window_size})")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)

        self.drop_path = DropPath(path_drop_prob) if (path_drop_prob > 0) else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwinMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_prob=drop_prob)
        self.windowing = SwinWindowing(window_size=window_size)

        self.H: Optional[int] = None  # placeholder
        self.W: Optional[int] = None  # placeholder

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        b, n, c = x.shape
        h, w = self.H, self.W
        if n != h * w:
            raise ValueError(f"Input shape {tuple(x.shape)} does not match with size ({h}, {w}).")

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b), mode=_SWIN_PADDING_MODE)
        _, hp, wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = self.windowing.window_partition(shifted_x)  # (b x num_windows, window_size, window_size, c)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # (~, window_size^2, c)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (~, window_size^2, c)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = self.windowing.window_reverse(attn_windows)  # (b, hp, wp, c)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if (pad_r > 0) or (pad_b > 0):
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            h: Spatial height of the input feature.
            w: Spatial width of the input feature.
        """
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"Input {tuple(x.shape)} does not match with size ({h}, {w})")

        x = x.view(b, h, w, c)

        # padding if not even
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, h % 2, 0, w % 2), mode=_SWIN_PADDING_MODE)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(b, -1, 4 * c)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float, optional): Dropout rate. Default: 0.0
        attn_drop_prob (float, optional): Attention dropout rate. Default: 0.0
        path_drop_prob (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim: int,
                 depth: int,
                 num_heads: int,
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: Union[float, Tuple[float, ...]] = 0.0,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_prob=drop_prob,
                attn_drop_prob=attn_drop_prob,
                path_drop_prob=path_drop_prob[i] if isinstance(path_drop_prob, (tuple, list)) else path_drop_prob,
            ) for i in range(depth)
        ])

        self.windowing = SwinWindowing(window_size=window_size)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            h: Spatial height of the input feature.
            w: Spatial width of the input feature.
        """

        # calculate attention mask for SW-MSA
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros(1, hp, wp, 1, dtype=x.dtype, device=x.device)  # (1, hp, wp, 1)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for _h in h_slices:
            for _w in w_slices:
                img_mask[:, _h, _w, :] = cnt
                cnt += 1

        mask_windows = self.windowing.window_partition(img_mask)  # (num_windows, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # if min(h, w) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     for blk in self.blocks:
        #         blk.shift_size = 0

        for blk in self.blocks:
            blk: SwinTransformerBlock
            blk.H, blk.W = h, w
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, h, w)
            h_down, w_down = (h + 1) // 2, (w + 1) // 2
            return x, h, w, x_down, h_down, w_down
        else:
            return x, h, w, x, h, w


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        out_norm (bool): Whether to use LayerNorm after patch embedding. Default: True
    """

    def __init__(self,
                 patch_size: Union[int, Tuple[int, int]] = 4,
                 in_channels: int = 3,
                 embed_dim: int = 96,
                 out_norm: bool = True):
        super().__init__()
        self.patch_size: Tuple[int, int] = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if out_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # padding
        _, _, h, w = x.size()

        if (h % self.patch_size[0] != 0) or (w % self.patch_size[1] != 0):
            pad_l = pad_t = 0
            pad_r = (self.patch_size[1] - w % self.patch_size[1]) % self.patch_size[1]
            pad_b = (self.patch_size[0] - h % self.patch_size[0]) % self.patch_size[0]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b), mode=_SWIN_PADDING_MODE)

        x = self.proj(x)  # (b, c, wh, ww)
        if self.norm is not None:
            wh, ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, wh, ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute position embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float): Dropout rate.
        attn_drop_prob (float): Attention dropout rate. Default: 0.
        path_drop_prob (float): Stochastic depth rate. Default: 0.2.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size: int = 224,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: float = 0.2,
                 ape: bool = False,
                 patch_norm: bool = True,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 frozen_stages: int = -1,
                 use_checkpoint: bool = False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim=embed_dim, out_norm=patch_norm)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = (pretrain_img_size, pretrain_img_size)
            patch_size = (patch_size, patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_prob)

        # stochastic depth
        pdp = [x.item() for x in torch.linspace(0, path_drop_prob, sum(depths))]  # stochastic depth decay rule

        # build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_prob=drop_prob,
                attn_drop_prob=attn_drop_prob,
                path_drop_prob=tuple(pdp[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = tuple(num_features)

        # force set last block NOT to shift - if input is 224
        # for blk in self.layers[-1].blocks:
        #     blk.shift_size = 0

        # add a norm layer for each output
        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            self.add_module(f"norm{i}", layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pretrained weights.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        if isinstance(pretrained, str):
            # using the default ckpt from original repository https://github.com/microsoft/Swin-Transformer
            state_dict = torch.load(pretrained, map_location="cpu")["model"]
            # remove unnecessary parameters
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if (k == "norm.weight") or (k == "norm.bias") or (k == "head.weight") or (k == "head.bias"):
                    continue
                elif "attn_mask" in k:
                    continue
                new_state_dict[k] = v
            for i in self.out_indices:
                new_state_dict[f"norm{i}.weight"] = getattr(self, f"norm{i}").weight.data.fill_(1.0)
                new_state_dict[f"norm{i}.bias"] = getattr(self, f"norm{i}").bias.data.fill_(0.0)
            self.load_state_dict(new_state_dict, strict=True)
        else:
            raise TypeError(f"Pretrained path should be string, got {pretrained}.")

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, ...]:
        """Forward function."""

        x = self.patch_embed(x)

        batch_size, _, wh, ww = x.shape
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(wh, ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2).contiguous()  # (b, wh * ww, c)
        else:
            x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, h, w, x, wh, ww = layer(x, wh, ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(batch_size, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                # outs.append(x_out)  # for DEBUG, used for "swin_transformer_check.py"

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freeze."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


if __name__ == '__main__':
    swin_base = SwinTransformer(
        224, patch_size=4, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=7)
    swin_base.init_weights(pretrained="checkpoint/swin_base_patch4_window7_224_22k.pth")
    swin_base_count = 0
    for p in swin_base.parameters():
        swin_base_count += p.numel()
    print(f"Swin Base parameters: {swin_base_count}")

    swin_large = SwinTransformer(
        224, patch_size=4, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), window_size=7)
    swin_large.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
    swin_large_count = 0
    for p in swin_large.parameters():
        swin_large_count += p.numel()
    print(f"Swin Large parameters: {swin_large_count}")
