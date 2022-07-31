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


class JIRANO(nn.Module):
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
        self.device = device
        self.vq = nn.Embedding(self.K, self.embedding_dim).cuda()
        self._vq_initialize()
        self.vq0_update = torch.zeros(self.K, device=self.device)

        if self.opt["initial"] == None:
            self._vq_initialized_from_batch = True
        else:
            self._vq_initialized_from_batch = False

    def _vq_initialize(self):  # TODO vq initialize
        initial_type = self.opt["initialize"].lower()
        if initial_type == "svd":
            pass
        elif initial_type == "tsp":
            pass
        elif initial_type == "uni":
            nn.init.xavier_uniform_(self.vq.weight)
        elif initial_type == "normal":
            nn.init.xavier_normal_(self.vq.weight)
            # nn.init.normal_(vq, std=0.1)
        else:
            raise ValueError(f"Unsupported vq initial type {initial_type}.")

    def _vector_quantize(self, feat: torch.Tensor, cur_iter) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param feat:            (batch_size, ch, h, w)
        :return:
                output feat:    (batch_size, ch, h, w)
                assignment:     (batch_size, K, h, w)
        """
        b, c, h, w = feat.shape
        # print("codebook", self.vq.weight)
        feat = feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        flat = feat.view(-1, self.embedding_dim)  # (bhw, c)

        if not self._vq_initialized_from_batch:
            with torch.no_grad():
                initialtype = self.opt["initial"]
                if initialtype == "rand":
                    random_select = list(range(b * h * w))
                    random.shuffle(random_select)
                    selected_indices = random_select[:self.K]
                    self.vq.weight.data.copy_(flat[selected_indices])

                elif initialtype == "kmeans":
                    cluster_ids_x, cluster_centers = kmeans(
                        X=flat, num_clusters=self.K, distance='euclidean', device=self.device
                    )
                    self.vq.weight.data.copy_(cluster_centers)
                else:
                    raise ValueError(f"Unsupported vq initial type {initialtype}.")
            self._vq_initialized_from_batch = True

        distance = (torch.sum(flat ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                    + torch.sum(self.vq.weight ** 2, dim=1)
                    - 2 * torch.matmul(flat, self.vq.weight.t()))  # (bhw, K)
        # print("distance", distance)
        # smaller distance == higher probability
        encodings = torch.zeros(distance.shape[0], self.K, device=flat.device)  # (bhw, K)
        encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)  # (bhw, 1)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector

        if self.opt["is_weight_sum"]:
            p = F.softmax(-distance, dim=1)
            q_feat = torch.matmul(p, self.vq.weight)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)
        else:
            q_feat = torch.matmul(encodings, self.vq.weight)  # (bhw, c)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

            # TODO -> vq_loss explode
            q_feat = feat + (q_feat - feat).detach()

        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        prob = torch.softmax(-distance, dim=1)  # (bhw, K)
        assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        # calculate update
        cur_update = torch.sum(encodings, dim=0)
        if self.training:
            self.vq0_update += cur_update

        return feat, q_feat, assignment, distance

    def forward(self, x: torch.Tensor, cur_iter, is_pos: bool = False):
        feat, x = self.extractor(x)  # Backbone (b, 384, h, w) -> Head : (b, d, h, w),
        if is_pos:
            return x, feat

        head, qx, assignment, distance = self._vector_quantize(x, cur_iter)  # (b, d, h, w), (b, K, h, w), (-1, K)
        if self.training:
            if cur_iter % 25 == 0:
                print("vq", torch.topk(self.vq0_update, 40).values,
                      torch.topk(self.vq0_update, 40, largest=False).values)
            # print(x.shape, qx.shape)
            return x, qx, assignment, distance, None, feat
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
