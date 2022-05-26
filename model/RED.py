from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .swin_transformer import SwinTransformer
from .red_swin_decoder import REDSwinDecoder


class RedSwinModel(nn.Module):

    def __init__(self,
                 dec_dim: int,  # 512
                 num_heads: int,  # 8
                 num_repeats: int,  # 3
                 num_emb: int,  # 128
                 window_size: int = 8,
                 encoder_type: str = "large",
                 output_scale: int = 4,
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 bias_type: str = "seg",
                 neck_type: str = "red"):
        super().__init__()

        swin_kwargs = dict(pretrain_img_size=224, patch_size=4, depths=(2, 2, 18, 2), window_size=7,
                           drop_prob=0.0, attn_drop_prob=0.0, path_drop_prob=0.2,
                           use_checkpoint=True)
        if (encoder_type == "base") or (encoder_type == "B"):
            swin = SwinTransformer(embed_dim=128, num_heads=(4, 8, 16, 32), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_base_patch4_window7_224_22k.pth")
        elif (encoder_type == "large") or (encoder_type == "L"):
            swin = SwinTransformer(embed_dim=192, num_heads=(6, 12, 24, 48), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
        else:
            raise ValueError(f"Unsupported SwinTransformer type {encoder_type}.")

        self.encoder = swin
        self.decoder = REDSwinDecoder(
            dec_dim,
            enc_dims=swin.num_features,  # (192, 384, 768, 1536)
            num_heads=num_heads,
            num_repeats=num_repeats,
            num_emb=num_emb,
            window_size=window_size,
            attn_drop_prob=attn_drop_prob,
            drop_prob=drop_prob,
            output_scale=output_scale,
            bias_type=bias_type,
            neck_type=neck_type,
        )
        self.num_repeats = num_repeats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        _, _, h, w = x.shape

        new_h = 448  # TODO fix hieght, width
        new_w = 672  # TODO fix hieght, width

        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

        features = self.encoder(x)
        outs, attn_weights = self.decoder(features)

        out = outs[-1]  # only use last one as inference output
        return out, outs, attn_weights

    @classmethod
    def build(cls, opt):
        # opt = opt["model"]
        m = cls(
            dec_dim=opt["dec_dim"],
            num_heads=opt["num_heads"],
            num_repeats=opt["num_repeats"],
            num_emb=opt["num_emb"],
            window_size=opt.get("window_size", 8),
            encoder_type=opt["encoder_type"],
            output_scale=opt.get("output_scale", 4),
            drop_prob=opt.get("drop_prob", 0.0),
            attn_drop_prob=opt.get("attn_drop_prob", 0.0),
            bias_type=opt.get("bias_type", "seg"),
            neck_type=opt.get("neck_type", "red"),
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = RedSwinModel(dec_dim=512, num_heads=8, num_repeats=3, num_emb=128, window_size=8, encoder_type="L")
    dummy_input = torch.empty(2, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (2, 1, 88, 304)
