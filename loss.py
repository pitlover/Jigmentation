from typing import Tuple, Dict
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


class StegoLoss(nn.Module):

    def __int__(self,
                n_classes: int,
                cfg: Dict,
                corr_weight: float = 1.0):
        super(StegoLoss, self).__init__()
        self.corr_loss = ContrastiveCorrelationLoss()
        self.linear_loss = LinearLoss()
        self.corr_weight = corr_weight
        self.n_classes = n_classes

    def forward(self, model_input, model_output, model_pos_output=None, linear_output: torch.Tensor() = None,
                cluster_output: torch.Tensor() = None) \
            -> Tuple[torch.Tensor, Dict[str, float]]:
        img, img_pos, label = model_input
        feats, code = model_output

        if self.corr_weight > 0:
            feats_pos, code_pos = model_pos_output
            corr_loss = self.corr_loss(feats, feats_pos, code, code_pos)
        else:
            corr_loss = torch.tensor(0, dtype=torch.float32, device=feats.device)

        linear_loss = self.linear_loss(linear_output, label, self.n_classes)
        cluster_loss = cluster_output[0]
        loss = (corr_loss * self.corr_weight) + linear_loss + cluster_loss
        loss_dict = {"loss": loss.item(), "corr": corr_loss.item(), "linear": linear_loss.item(),
                     "cluster": cluster_loss.item()}

        return loss, loss_dict


class ContrastiveCorrelationLoss(nn.Module):  # TODO need to analysis

    def __init__(self, cfg):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (self.cfg["pos_inter_weight"] * pos_inter_loss +
                self.cfg["pos_intra_weight"] * pos_intra_loss +
                self.cfg["neg_inter_weight"] * neg_inter_loss)


class LinearLoss(nn.Module):

    def __init__(self, cfg):
        super(LinearLoss, self).__init__()
        self.cfg = cfg
        self.linear_loss = nn.CrossEntropyLoss()

    def forward(self, linear_logits: torch.Tensor, label: torch.Tensor, n_classes: int):
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        linear_loss = self.linear_loss(linear_logits[mask], flat_label[mask]).mean()

        return linear_loss
