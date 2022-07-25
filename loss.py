from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class HoiLoss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 batch_size: int,
                 cfg: Dict):
        super().__init__()
        self.opt = cfg
        self.n_classes = n_classes

        self.vq_weight = self.opt["vq_weight"]
        self.cross_weight = self.opt["cross_weight"]

        self.vq_loss = VQLoss(cfg["vq_loss"])
        self.cross_loss = CrossLoss(cfg["cross_loss"], batch_size)

        self.linear_loss = LinearLoss(cfg)

    def forward(self, model_input: Tuple,
                model_output: Tuple,
                model_pos_output: Tuple,
                linear_output: torch.Tensor = None,
                cluster_output: torch.Tensor = None):
        loss, loss_dict, vq_dict = 0, {}, {}

        if self.vq_weight > 0:
            vq_loss, vq_dict = self.vq_loss(model_output)
            loss += self.vq_weight * vq_loss
            loss_dict["vq"] = vq_loss.item()

        if self.cross_weight > 0:
            cross_loss = self.cross_loss(model_output, model_input)
            loss += self.cross_weight * cross_loss
            loss_dict["cross"] = cross_loss.item()

        linear_loss = self.linear_loss(linear_output, model_input[1], self.n_classes)
        cluster_loss = cluster_output[0]

        loss_dict["loss"], loss_dict["linear"], loss_dict[
            "cluster"] = loss.item(), linear_loss.item(), cluster_loss.item()

        return loss, loss_dict, vq_dict


class CrossLoss(nn.Module):
    def __init__(self, cfg: Dict, batch_size: int):
        super().__init__()
        self.opt = cfg
        self.temperature = self.opt["temperature"]
        self.batch_size = batch_size

        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.ce = torch.nn.CrossEntropyLoss(reduction="sum")
        self.get_corr_mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        # diag = np.eye(2 * self.batch_size)
        # l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # mask = torch.from_numpy((diag + l1 + l2))
        # mask = (1 - mask).type(torch.bool)
        # return mask.cuda()

        diag = np.eye(2 * self.batch_size * 70)
        l1 = np.eye((2 * self.batch_size * 70), 2 * self.batch_size * 70, k=-self.batch_size * 70)
        l2 = np.eye((2 * self.batch_size * 70), 2 * self.batch_size * 70, k=self.batch_size * 70)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    def forward(self, model_output: Tuple, model_input: Tuple):
        # head, quantized, assignment, distance = model_output  # [x1, x2], [qx1, qx2], [ass1, ass2], [dis1, dis2]
        # # head (b, d, h, w)
        # b, d, h, w = head[0].shape
        #
        # x1 = head[0].reshape(self.batch_size, -1)  # (b, dhw)
        # x2 = head[1].reshape(self.batch_size, -1)  # (b, dhw)
        #
        # z1 = quantized[0].reshape(self.batch_size, -1)  # (b, dhw)
        # z2 = quantized[1].reshape(self.batch_size, -1)  # (b, dhw)
        #
        # x1z2 = torch.cat([x1, z2], dim=0)  # (2b, -1) -> (2b, dhw)
        # x2z1 = torch.cat([x2, z1], dim=0)  # (2b, -1) -> (2b, dhw)
        #
        # '''
        # cos | x1z1 | x1z3 | x1z2 | x1z4
        # --------------------------------
        # x1z1|  1  |  N  |  P  |  N  |
        # x1z3|  N  |  1  |  N  |  P  |
        # x1z2|  P  |  N  |  1  |  N  |
        # x1z4|  N  |  P  |  N  |  1  |
        # '''
        # cos_12 = self.cos(x1z2.unsqueeze(1), x1z2.unsqueeze(0))  # (2b, 2b)
        # Right_12 = torch.diagonal(cos_12, self.batch_size)
        # Left_12 = torch.diagonal(cos_12, -self.batch_size)
        # pos_12 = torch.cat([Right_12, Left_12]).view(2 * self.batch_size, 1)
        # neg_12 = cos_12[self.get_corr_mask].view(2 * self.batch_size, -1)
        #
        # cos_21 = self.cos(x2z1.unsqueeze(1), x2z1.unsqueeze(0))
        # Right_21 = torch.diagonal(cos_21, self.batch_size)
        # Left_21 = torch.diagonal(cos_21, -self.batch_size)
        # pos_21 = torch.cat([Right_21, Left_21]).view(2 * self.batch_size, 1)
        # neg_21 = cos_21[self.get_corr_mask].view(2 * self.batch_size, -1)
        #
        # logits_12 = torch.cat((pos_12, neg_12), dim=1)     # (2b, 2b-1)
        # logits_12 /= self.temperature
        #
        # logits_21 = torch.cat((pos_21, neg_21), dim=1)
        # logits_21 /= self.temperature
        #
        # labels = torch.zeros(2 * self.batch_size).to(head[0].device).long()
        # cross_loss = self.ce(logits_12, labels) + self.ce(logits_21, labels)
        #
        # return cross_loss / (2 * self.batch_size)

        head, quantized, assignment, distance = model_output  # [x1, x2], [qx1, qx2], [ass1, ass2], [dis1, dis2]
        # head (b, d, h, w)
        b, d, h, w = head[0].shape

        x1 = head[0].reshape(self.batch_size * d, -1)  # (bd, hw)
        x2 = head[1].reshape(self.batch_size * d, -1)  # (bd, hw)

        z1 = quantized[0].reshape(self.batch_size * d, -1)  # (bd, hw)
        z2 = quantized[1].reshape(self.batch_size * d, -1)  # (bd, hw)

        x1z2 = torch.cat([x1, z2], dim=0)  # (2bd, hw)
        x2z1 = torch.cat([x2, z1], dim=0)  # (2bd, hw)

        '''
        cos | x1z1 | x1z3 | x1z2 | x1z4
        --------------------------------
        x1z1|  1  |  N  |  P  |  N  |
        x1z3|  N  |  1  |  N  |  P  |
        x1z2|  P  |  N  |  1  |  N  |
        x1z4|  N  |  P  |  N  |  1  |
        '''
        cos_12 = self.cos(x1z2.unsqueeze(1), x1z2.unsqueeze(0))  # (2bd, 2bd)

        Right_12 = torch.diagonal(cos_12, self.batch_size * d)  # (2bd - b)
        Left_12 = torch.diagonal(cos_12, -self.batch_size * d)  # (2bd - b)

        pos_12 = torch.cat([Right_12, Left_12]).view(2 * self.batch_size * d, 1)  # (2bd, 1)
        neg_12 = cos_12[self.get_corr_mask].view(2 * self.batch_size * d, -1)  # (2bd, 2(bd-1))

        cos_21 = self.cos(x2z1.unsqueeze(1), x2z1.unsqueeze(0))
        Right_21 = torch.diagonal(cos_21, self.batch_size * d)
        Left_21 = torch.diagonal(cos_21, -self.batch_size * d)
        pos_21 = torch.cat([Right_21, Left_21]).view(2 * self.batch_size * d, 1)
        neg_21 = cos_21[self.get_corr_mask].view(2 * self.batch_size * d, -1)

        logits_12 = torch.cat((pos_12, neg_12), dim=-1)
        logits_12 /= self.temperature

        logits_21 = torch.cat((pos_21, neg_21), dim=-1)
        logits_21 /= self.temperature

        labels = torch.zeros(2 * self.batch_size * d).to(head[0].device).long()
        cross_loss = self.ce(logits_12, labels) + self.ce(logits_21, labels)

        return cross_loss / (2 * self.batch_size * d)


class VQLoss(nn.Module):
    def __init__(self,
                 cfg: dict
                 ):
        super().__init__()
        self.opt = cfg
        self.manage_weight = self.opt["manage_weight"]

    def forward(self, model_output: Tuple):
        head, quantized, assignment, distance = model_output  # [x1, x2], [qx1, qx2], [ass1, ass2], [dis1, dis2]
        vq_dict = {}
        q_list, e_list, inter_list, loss = [], [], [], 0

        for a in range(len(head)):
            # ch = head[a].shape[1]
            q_loss = F.mse_loss(head[a].detach(), quantized[a])
            e_loss = F.mse_loss(quantized[a], head[a].detach())
            q_list.append(q_loss)
            e_list.append(e_loss)

            # manageloss
            if self.manage_weight > 0.0:
                p = F.softmax(distance[a], dim=1)
                entropy = -p * torch.log(p + 1e-8)
                entropy = torch.sum(entropy, dim=-1)  # (25088,)
                intra_loss = entropy.mean()  # minimization

                avg_p = p.mean(0)
                avg_entropy = -avg_p * torch.log(avg_p + 1e-8)
                avg_entropy = torch.sum(avg_entropy, dim=-1)
                inter_loss = -avg_entropy  # maximization
                inter_list.append(inter_loss)

                if self.opt["intra_weight"] > 0.0:
                    vq_dict[f"[{a}]intra"] = intra_loss
                if self.opt["inter_weight"] > 0.0:
                    vq_dict[f"[{a}]inter"] = inter_loss

                loss += (self.opt["intra_weight"] * intra_loss + self.opt["inter_weight"] * inter_loss)

            loss += (self.opt["q_weight"] * q_loss + self.opt["e_weight"] * e_loss)

            vq_dict[f"[{a}]e_loss"] = e_loss
            vq_dict[f"[{a}]q_loss"] = q_loss

        vq_dict.update({"e_vq": sum(e_list), "q_vq": sum(q_list), "loss": loss})

        return loss, vq_dict


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


class StegoLoss(nn.Module):

    def __init__(self,
                 n_classes: int,
                 cfg: dict,
                 corr_weight: float = 1.0):
        super().__init__()

        self.n_classes = n_classes
        self.corr_weight = corr_weight
        self.corr_loss = ContrastiveCorrelationLoss(cfg)
        self.linear_loss = LinearLoss(cfg)

    def forward(self, model_input, model_output, model_pos_output=None, linear_output: torch.Tensor() = None,
                cluster_output: torch.Tensor() = None) \
            -> Tuple[torch.Tensor, Dict[str, float]]:
        img, label = model_input
        feats, code = model_output

        if self.corr_weight > 0:
            feats_pos, code_pos = model_pos_output
            corr_loss, corr_loss_dict = self.corr_loss(feats, feats_pos, code, code_pos)
        else:
            corr_loss_dict = {"none": 0}
            corr_loss = torch.tensor(0, dtype=torch.float32, device=feats.device)

        linear_loss = self.linear_loss(linear_output, label, self.n_classes)
        cluster_loss = cluster_output[0]
        loss = (corr_loss * self.corr_weight) + linear_loss + cluster_loss
        loss_dict = {"loss": loss.item(), "corr": corr_loss.item(), "linear": linear_loss.item(),
                     "cluster": cluster_loss.item()}

        return loss, loss_dict, corr_loss_dict


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg["pointwise"]:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg["zero_clamp"]:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg["stabilize"]:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor,
                orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg["feature_samples"], self.cfg["feature_samples"], 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg["corr_loss"]["pos_intra_shift"])
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg["corr_loss"]["pos_inter_shift"])

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg["neg_samples"]):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg["corr_loss"]["neg_inter_shift"])
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (self.cfg["corr_loss"]["pos_intra_weight"] * pos_intra_loss.mean() +
                self.cfg["corr_loss"]["pos_inter_weight"] * pos_inter_loss.mean() +
                self.cfg["corr_loss"]["neg_inter_weight"] * neg_inter_loss.mean(),
                {"self_loss": pos_intra_loss.mean().item(),
                 "knn_loss": pos_inter_loss.mean().item(),
                 "rand_loss": neg_inter_loss.mean().item()}
                )


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
