from typing import List, Tuple, Any, Optional
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import kornia.augmentation as Kg
from utils.layer_utils import ClusterLookup
from model.dino.DinoFeaturizer import DinoFeaturizer
import torchvision.transforms as transforms
import numpy as np
from kmeans_pytorch import kmeans


class BOB(nn.Module):
    # opt["model"]
    def __init__(self,
                 opt: dict,
                 n_classes: int,
                 device: torch.device
                 ):
        super().__init__()
        self.opt = opt
        self.n_classes = n_classes

        if not opt["continuous"]:
            dim = n_classes
        else:
            dim = opt["dim"]

        if opt["arch"] == "dino":
            self.extractor = DinoFeaturizer(dim, opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.K = opt["K"]
        self.embedding_dim = opt["dim"]
        self.normalize = opt["normalize"]
        self.device = device

        self.out_channel = self.opt["num_feats"]
        # TODO Bob decoder
        self.decoder = BOB_decoder(
            opt=self.opt["decoder"],
            in_channel=self.embedding_dim,
            out_channel=self.out_channel
        )

        self.vq = nn.Parameter(torch.rand(self.K, dim), requires_grad=True)
        self._vq_initialize()

        if self.opt["initial"] == None:
            self._vq_initialized_from_batch = True
        else:
            self._vq_initialized_from_batch = False

        self.vq0_update = torch.zeros(self.K, device=self.device)
        self.restart = self.opt["restart"]
        self.num_update = 0
        # self.dropout = nn.Dropout(p=0.3)

    def _get_temperature(self, cur_iter):
        # TODO get temperature
        # return max((1000 - cur_iter) / 1000, 0.1)  # 0 -> 1, 900 -> 0.1
        return self.opt["temperature"]

    def _get_gate(self, cur_iter):
        # TODO get gate
        return 0.0
        # return max((10000 - cur_iter) / 10000, 0.0)  # 0 -> 1, 1000 -> 0

    def _vq_initialize(self):  # TODO vq initialize
        initial_type = self.opt["initialize"].lower()
        if initial_type == "svd":
            pass
        elif initial_type == "tsp":
            pass
        elif initial_type == "uni":
            nn.init.xavier_uniform_(self.vq)
        elif initial_type == "normal":
            nn.init.xavier_normal_(self.vq)
        else:
            raise ValueError(f"Unsupported vq initial type {initial_type}.")

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor, cur_iter: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param feat:            (batch_size, ch, h, w)
        :param codebook:        (K, ch)
        :return:
                output feat:    (batch_size, ch, h, w)
                assignment:     (batch_size, K, h, w)
        """
        b, c, h, w = feat.shape
        k, _ = codebook.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        flat = feat.view(-1, c)  # (bhw, c)

        if self.normalize:
            flat = F.normalize(flat, dim=1)  # v / ||v||

        # Z-normalize
        # flat_var, flat_mean = torch.var_mean(flat, dim=1, keepdim=True)
        # flat_inv_std = torch.rsqrt(flat_var + 1e-6)
        # flat = (flat - flat_mean) * flat_inv_std

        if not self._vq_initialized_from_batch:
            with torch.no_grad():
                initialtype = self.opt["initial"]
                if initialtype == "rand":
                    random_select = list(range(b * h * w))
                    random.shuffle(random_select)
                    selected_indices = random_select[:k]
                    codebook.data.copy_(flat[selected_indices])

                elif initialtype == "kmeans":
                    cluster_ids_x, cluster_centers = kmeans(
                        X=flat, num_clusters=k, distance='euclidean', device=self.device
                    )
                    codebook.data.copy_(cluster_centers)
                else:
                    raise ValueError(f"Unsupported vq initial type {initialtype}.")
            self._vq_initialized_from_batch = True

        code = codebook  # (K, c)

        # code = self.dropout(code)
        if self.normalize:
            code = F.normalize(codebook, dim=1)

        distance = (torch.sum(flat ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                    + torch.sum(code ** 2, dim=1)
                    - 2 * torch.matmul(flat, code.t()))  # (bhw, K)

        # smaller distance == higher probability
        encodings = torch.zeros(distance.shape[0], self.K, device=flat.device)  # (bhw, K)
        encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)  # (bhw, 1)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector

        if self.opt["is_weight_sum"]:
            p = F.softmax(-distance / self._get_temperature(cur_iter), dim=1)
            q_feat = torch.matmul(p, code)  # (bhw, K) x (K, c) = (bhw, c)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

        else:  # TODO top1
            q_feat = torch.matmul(encodings, code)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

            # TODO -> vq_loss explode
            q_feat = feat + (q_feat - feat).detach()

        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        prob = torch.softmax(-distance / self._get_temperature(cur_iter), dim=1)  # (bhw, K)
        assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        # calculate update
        if self.training:
            cur_update = torch.sum(encodings, dim=0)
            self.vq0_update += cur_update
            # TODO restart

        return feat, q_feat, assignment, distance

    def forward(self, x: torch.Tensor, cur_iter, is_pos: bool = False):
        feat, x = self.extractor(x)  # Backbone (b, 384, h, w) -> Head : (b, d, h, w),
        if is_pos:
            return x, feat

        head, qx, assignment, distance = self._vector_quantize(x, self.vq,
                                                               cur_iter)  # (b, d, h, w), (b, K, h, w), (-1, K)
        if self.training:
            recon = self.decoder(qx)
            if cur_iter % 25 == 0:
                print("vq", torch.topk(self.vq0_update, 40).values,
                      torch.topk(self.vq0_update, 40, largest=False).values)
            return x, qx, assignment, distance, recon, feat
        else:
            return x, qx

    @classmethod
    def build(cls, opt, n_classes, device):
        # opt = opt["model"]
        m = cls(
            opt=opt,
            n_classes=n_classes,
            device=device
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


class BOB_decoder(nn.Module):
    # opt["decoder"]
    def __init__(self,
                 opt,
                 in_channel: int,
                 out_channel: int
                 ):
        super().__init__()
        self.opt = opt
        n_res_block = self.opt["n_res_block"]
        n_res_channel = self.opt["n_res_channel"]
        channel = self.opt["channel"]
        self.prenorm = self.opt["prenorm"]

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        # TODO layernorm dimensin right?
        self.norm = nn.LayerNorm(out_channel, eps=1e-6)
        self.final = nn.Conv2d(channel, out_channel, 1)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, qx: torch.Tensor) -> torch.Tensor:
        # qx : (b, d, h, w)
        self.shape = qx.shape
        x = self.blocks(qx)

        if self.prenorm:
            x = self.norm(x)
            x = self.final(x)
        else:
            x = self.final(x)  # (b, c, h, w)
            x = self.norm(x.permute(0, 2, 3, 1))  # (b, h, w, c)
            x = x.permute(0, 3, 1, 2)  # (b, c, h, w)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 3, padding=1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
