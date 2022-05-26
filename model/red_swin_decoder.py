from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .utils import ConvBN, _CONV_PADDING_MODE
from .swin_transformer import SwinWindowing


class PreNormFF(nn.Module):

    def __init__(self,
                 in_dims: int,
                 drop_prob: float = 0.0,
                 feedforward_dims: Optional[int] = None,
                 act_layer=nn.GELU):
        super().__init__()
        if feedforward_dims is None:
            feedforward_dims = 4 * in_dims
        self.in_dims = in_dims

        self.norm = nn.LayerNorm(in_dims)
        self.lin1 = nn.Linear(in_dims, feedforward_dims)
        self.lin2 = nn.Linear(feedforward_dims, in_dims)
        self.drop = nn.Dropout(drop_prob)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)

        x = x + identity
        return x


class PreNormDWConvFF(nn.Module):

    def __init__(self,
                 in_dims: int,
                 drop_prob: float = 0.0,
                 feedforward_dims: Optional[int] = None,
                 kernel_size: int = 5,
                 act_layer=nn.GELU):
        super().__init__()
        if feedforward_dims is None:
            feedforward_dims = 4 * in_dims
        self.in_dims = in_dims
        self.feedforward_dims = feedforward_dims

        self.norm = nn.LayerNorm(in_dims)
        self.lin1 = nn.Linear(in_dims, feedforward_dims * 2)
        self.act1 = nn.GLU(dim=-1)

        self.kernel_size = kernel_size
        self.conv2 = nn.Conv2d(feedforward_dims, feedforward_dims, kernel_size=(kernel_size, kernel_size), bias=False,
                               stride=(1, 1), padding=(2, 2), padding_mode=_CONV_PADDING_MODE, groups=feedforward_dims)
        self.bn2 = nn.BatchNorm2d(feedforward_dims)
        self.act2 = act_layer()

        self.lin3 = nn.Linear(feedforward_dims, in_dims)
        self.drop = nn.Dropout(drop_prob)

        self.initialize_parameters()

    def initialize_parameters(self):
        kernel_size = self.conv2.kernel_size
        nn.init.normal_(self.conv2.weight, mean=0.0, std=math.sqrt(2 / (kernel_size[0] * kernel_size[1])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)
        x = self.lin1(x)
        x = self.act1(x)

        x = x.permute(0, 3, 1, 2)  # (n, h, w, c) -> (n, c, h, w)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = x.permute(0, 2, 3, 1)  # (n, c, h, w) -> (n, h, w, c)

        x = self.lin3(x)
        x = self.drop(x)

        x = x + identity
        return x


class REDSwinSA(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_emb: int,
                 window_size: int = 8,
                 shift_size: int = 0,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 bias_type: str = "seg"):
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads
        if in_dims % num_heads != 0:
            raise ValueError(f"Input dim {in_dims} is not divisible by num_heads {num_heads}.")
        self.head_dim = in_dims // num_heads

        self.norm = nn.LayerNorm(in_dims)
        self.q_proj = nn.Linear(in_dims, in_dims)
        self.k_proj = nn.Linear(in_dims, in_dims)
        self.v_proj = nn.Linear(in_dims, in_dims)
        self.o_proj = nn.Linear(in_dims, in_dims)

        self.attn_scale = math.sqrt(1 / self.head_dim)
        self.drop = nn.Dropout(drop_prob)
        self.attn_drop = nn.Dropout(attn_drop_prob)

        self.window_size = window_size
        self.shift_size = shift_size
        self.windowing = SwinWindowing(window_size=window_size)
        assert (self.window_size == 8) or (self.window_size == 4)

        self.num_emb = num_emb
        self.bias_type = bias_type
        if bias_type == "seg":
            with torch.no_grad():
                depth_embedding = torch.linspace(1, 2 * num_emb - 1, 2 * num_emb - 1)  # (2n-1,) range [1, 2n-1]
                depth_embedding -= num_emb  # range [-(n-1), n-1]
                depth_embedding = depth_embedding.unsqueeze(-1).expand(2 * num_emb - 1, num_heads)  # (2n - 1, nd)
                depth_embedding_init = torch.ones(num_heads, dtype=torch.float32).uniform_(0.01, 0.04)  # noqa
                depth_embedding = depth_embedding.contiguous()
                depth_embedding[:num_emb] *= depth_embedding_init
                depth_embedding[-num_emb:] *= (-depth_embedding_init)
            self.depth_embedding = nn.Parameter(depth_embedding, requires_grad=True)
        elif bias_type == "none":
            pass
        elif bias_type == "pos":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported bias type {bias_type}.")

    def _reshape_4d(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        x = x.view(b, h * w, self.num_heads, d // self.num_heads).transpose(1, 2)
        return x.contiguous()

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = x.shape
        assert d == self.in_dims
        r = self.window_size

        identity = x

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            indices = torch.roll(indices, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        if self.bias_type == "seg":
            indices = self.windowing.window_partition(indices.unsqueeze(-1))  # (b, h, w, 1) -> (*, r, r, 1)
            rel = indices.view(-1, r * r, 1) - indices.view(-1, 1, r * r)  # (*, rr, rr)  range [-(n-1), n-1]
            rel += (self.num_emb - 1)  # range [0, 2n-2]
            de = F.embedding(rel, weight=self.depth_embedding)  # (*, rr, rr, nh)
            de = de.permute(0, 3, 1, 2)

        elif self.bias_type == "none":
            de = 0

        elif self.bias_type == "pos":
            raise NotImplementedError

        else:
            raise ValueError  # shouldn't be here

        x = self.windowing.window_partition(x)  # (b, h, w, d) -> (*, r, r, d)

        x_norm = self.norm(x)

        q = self.q_proj(x_norm)  # (*, r, r, d)
        k = self.k_proj(x_norm)  # (*, r, r, d)
        v = self.v_proj(x_norm)  # (*, r, r, d)

        q_flat = self._reshape_4d(q)  # (*, nh, rr, hd)
        k_flat = self._reshape_4d(k)  # (*, nh, rr, hd)
        v_flat = self._reshape_4d(v)  # (*, nh, rr, hd)

        attn = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (*, nh, rr, rr)
        attn *= self.attn_scale
        attn = self.attn_drop(attn)  # Here, we only drop attention BEFORE softmax and de addition.
        attn = attn + de
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v_flat)  # (*, nh, rr, hd)
        out = out.transpose(1, 2).reshape(-1, r, r, d)
        out = self.o_proj(out)
        out = self.drop(out)

        out = self.windowing.window_reverse(out)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        out = out + identity
        return out, attn


class REDSwinBlock(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_emb: int,
                 window_size: int = 8,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU,
                 bias_type: str = "seg"):
        super().__init__()

        sa_kwargs = dict(window_size=window_size, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob,
                         bias_type=bias_type)
        ff_kwargs = dict(feedforward_dims=feedforward_dims, drop_prob=drop_prob, act_layer=act_layer)

        self.sa1 = REDSwinSA(in_dims, num_heads, num_emb, shift_size=0, **sa_kwargs)
        self.ff1 = PreNormDWConvFF(in_dims, **ff_kwargs)

        self.sa2 = REDSwinSA(in_dims, num_heads, num_emb, shift_size=window_size // 2, **sa_kwargs)
        self.ff2 = PreNormDWConvFF(in_dims, **ff_kwargs)

        self.linear = nn.Linear(in_dims, in_dims, bias=False)
        self.norm = nn.LayerNorm(in_dims, elementwise_affine=True)

    def forward(self, x: torch.Tensor,
                indices: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # b, h, w, c = x.shape

        x, attn1 = self.sa1(x, indices)
        x = self.ff1(x)

        x, attn2 = self.sa2(x, indices)
        x = self.ff2(x)

        x = self.linear(x)
        x = self.norm(x)

        return x, (attn1, attn2)


class REDSwinHead(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_repeats: int,
                 num_emb: int = 128,
                 window_size: int = 8,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU,
                 bias_type: str = "seg"):
        super().__init__()
        self.in_dims = in_dims
        self.num_repeats = num_repeats
        self.num_emb = num_emb

        conv_kwargs = dict(act_layer=act_layer)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                ConvBN(in_dims, in_dims // 4, 3, **conv_kwargs),
                ConvBN(in_dims // 4, in_dims // 4, 3, **conv_kwargs),
                nn.Conv2d(in_dims // 4, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            ) for _ in range(num_repeats)
        ])

        self.conv_layers.append(
            nn.Sequential(
                ConvBN(in_dims, in_dims // 4, 3, **conv_kwargs),
                ConvBN(in_dims // 4, in_dims // 4, 3, **conv_kwargs),
                nn.Conv2d(in_dims // 4, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
        )

        self.attn_layers = nn.ModuleList([
            REDSwinBlock(in_dims, num_heads, num_emb, window_size, feedforward_dims=feedforward_dims,
                         attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer,
                         bias_type=bias_type)
            for _ in range(num_repeats)
        ])

        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def _logit_to_indices(self, out: torch.Tensor) -> torch.Tensor:  # uniform split
        # b, c, h, w = out.shape
        assert out.shape[1] == 1  # 1-channel
        indices = self.sigmoid(out.detach())  # (0, 1)
        indices = torch.floor(indices * self.num_emb - 1e-3)  # [0, 128)
        indices = indices.long().squeeze(1)  # (b, h, w)
        return indices

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # b, h, w, c = x.shape

        outs = list()
        attn_weights = tuple()

        for i in range(self.num_repeats):
            x_chw = x.permute(0, 3, 1, 2)
            logit = self.conv_layers[i](x_chw)  # (b, 1, h, w)
            outs.append(logit)

            indices = self._logit_to_indices(logit)

            x, aws = self.attn_layers[i](x, indices)
            attn_weights += aws

        x_chw = x.permute(0, 3, 1, 2)
        logit = self.conv_layers[-1](x_chw)  # (b, 1, h, w)
        outs.append(logit)

        outs = tuple(outs)
        return outs, attn_weights


class REDSwinDecoder(nn.Module):

    def __init__(self,
                 dec_dim: int = 512,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 num_heads: int = 8,
                 num_repeats: int = 3,
                 num_emb: int = 128,
                 window_size: int = 8,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 output_scale: int = 4,
                 act_layer=nn.GELU,
                 bias_type: str = "seg",
                 neck_type: str = "red"):
        super().__init__()

        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4
        if dec_dim % 4 != 0:
            raise ValueError(f"Decoder dim {dec_dim} should be a multiple of 4.")

        # -------------------------------------------------------------- #
        # Neck
        # -------------------------------------------------------------- #

        conv_kwargs = dict(act_layer=act_layer)

        self.neck_type = neck_type
        if neck_type == "red":
            self.enc_conv32 = nn.Sequential(
                ConvBN(enc_dims[3], dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=8)
            )
            self.enc_conv16 = nn.Sequential(
                ConvBN(enc_dims[2], dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=4)
            )
            self.enc_conv8 = nn.Sequential(
                ConvBN(enc_dims[1], dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.enc_conv4 = nn.Sequential(
                ConvBN(enc_dims[0], dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.Identity()  # for consistency
            )
            self.enc_fuse = ConvBN(dec_dim * 4, dec_dim, kernel_size=1, act_layer=act_layer)
            enc_channels = dec_dim

        elif neck_type == "fpn":
            self.enc_conv32 = nn.Sequential(
                ConvBN(enc_dims[3], dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.enc_conv16 = nn.Sequential(
                ConvBN(enc_dims[2] + dec_dim, dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.enc_conv8 = nn.Sequential(
                ConvBN(enc_dims[1] + dec_dim, dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.enc_conv4 = nn.Sequential(
                ConvBN(enc_dims[0] + dec_dim, dec_dim, 3, **conv_kwargs),
                ConvBN(dec_dim, dec_dim, 3, **conv_kwargs),
                nn.Identity()  # for consistency
            )
            enc_channels = dec_dim

        else:
            raise ValueError(f"Unsupported neck type {neck_type}.")

        self.dec_linear = nn.Linear(enc_channels, dec_dim, bias=False)
        self.dec_norm = nn.LayerNorm(dec_dim, elementwise_affine=True)

        # -------------------------------------------------------------- #
        # Head
        # -------------------------------------------------------------- #

        self.reducer = REDSwinHead(
            dec_dim, num_heads, num_repeats, num_emb=num_emb, window_size=window_size,
            attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer, bias_type=bias_type,
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            # zero filling biases
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) and (module.bias is not None):
                nn.init.zeros_(module.bias)

    def forward(self, enc_features: torch.Tensor
                ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward function."""
        e4, e8, e16, e32 = enc_features

        if self.neck_type == "red":
            e32 = self.enc_conv32(e32)
            e16 = self.enc_conv16(e16)
            e8 = self.enc_conv8(e8)
            e4 = self.enc_conv4(e4)

            # 1/4 scale
            dec = torch.cat([e4, e8, e16, e32], dim=1)  # (b, c, h, w)
            dec = self.enc_fuse(dec)
            dec = dec.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)

        elif self.neck_type == "fpn":
            e32 = self.enc_conv32(e32)
            e16 = self.enc_conv16(torch.cat([e16, e32], dim=1))
            e8 = self.enc_conv8(torch.cat([e8, e16], dim=1))
            e4 = self.enc_conv4(torch.cat([e4, e8], dim=1))

            dec = e4.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)

        else:
            raise ValueError  # shouldn't be here

        dec = self.dec_linear(dec)
        dec = self.dec_norm(dec)

        outs, attn_weights = self.reducer(dec)

        return outs, attn_weights
