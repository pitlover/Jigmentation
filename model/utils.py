from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

_CONV_PADDING_MODE = "replicate"


class ConvBN(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int,
                 conv_groups: int = 1,
                 act_layer: Optional = nn.GELU):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode=_CONV_PADDING_MODE,
            groups=conv_groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer() if (act_layer is not None) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
