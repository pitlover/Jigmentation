from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class REDloss(nn.Module):

    def __init__(self, loss_opt, ignore_label):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.ce_weight = loss_opt['ce_weight']

    def forward(self, model_input, model_output) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, gt_target = model_input
        prediction, attn_weights = model_output

        ce_loss = self.ce(prediction, gt_target)

        loss = (ce_loss * self.si_weight)
        loss_dict = {"loss": loss.item(), "ce": ce_loss.item()}

        return loss, loss_dict
