from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


# TODO Dulli Loss
class DulliLoss(nn.Module):
    def __init__(self,
                 cfg: dict
                 ):
        super().__init__()
        self.opt = cfg

    def forward(self, model_input, model_output: Tuple):
        head, quantized, assignment, distance = model_output  # (x0, x1, x2, x3), (vq0, vq1, vq2, vq3)
        loss_dict = {}
        q_list, e_list, inter_list, loss = [], [], [], 0

        for a in range(len(head)):
            # ch = head[a].shape[1]
            q_loss = F.mse_loss(head[a].detach(), quantized[a])
            e_loss = F.mse_loss(quantized[a], head[a].detach())
            q_list.append(q_loss)
            e_list.append(e_loss)

            # manageloss
            p = F.softmax(distance[a], dim=1)
            entropy = -p * torch.log(p + 1e-8)
            entropy = torch.sum(entropy, dim=-1)  # (25088,)
            intra_loss = entropy.mean()  # minimization

            avg_p = p.mean(0)
            avg_entropy = -avg_p * torch.log(avg_p + 1e-8)
            avg_entropy = torch.sum(avg_entropy, dim=-1)
            inter_loss = -avg_entropy
            inter_list.append(inter_loss)

            # TODO need to fix
            # if a == 1:
            #     sub_loss = ( (q_loss + self.opt["e_weight"] * e_loss) + self.opt["inter_weight"] * inter_loss)
            #     loss += sub_loss
            # else:
            #     sub_loss = (q_loss + self.opt["e_weight"] * e_loss + self.opt["inter_weight"] * inter_loss)
            #     loss += sub_loss
            # loss += (q_loss + self.opt["e_weight"] * e_loss + self.opt["intra_weight"] * intra_loss + self.opt["inter_weight"] * inter_loss)
            # loss += (q_loss + self.opt["e_weight"] * e_loss + self.opt["inter_weight"] * inter_loss)  # SWaV style
            # loss += (q_loss + self.opt["e_weight"] * e_loss + self.opt["intra_weight"] * intra_loss)  # my intuition
            loss += (q_loss + self.opt["e_weight"] * e_loss)

            loss_dict[f"[{a}]e_loss"] = e_loss
            loss_dict[f"[{a}]q_loss"] = q_loss
            loss_dict[f"[{a}]inter_loss"] = inter_loss
            # loss_dict[f"[{a}]total_loss"] = sub_loss

        loss_dict.update({"e_vq": sum(e_list), "q_vq": sum(q_list), "inter": sum(inter_list), "loss": loss})

        return loss, loss_dict


class LinearLoss(nn.Module):

    def __init__(self, cfg: dict):
        super(LinearLoss, self).__init__()
        self.cfg = cfg
        self.linear_loss = nn.CrossEntropyLoss()

    def forward(self, linear_logits: torch.Tensor, label: torch.Tensor, n_classes: int):
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < n_classes)

        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        linear_loss = self.linear_loss(linear_logits[mask], flat_label[mask]).mean()

        return linear_loss
