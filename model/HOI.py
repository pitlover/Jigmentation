from typing import List, Tuple, Any, Optional
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import kornia.augmentation as Kg
from utils.layer_utils import ClusterLookup
from model.dino.DinoFeaturizer import DinoFeaturizer
from model.res.fpn import Resnet
import torchvision.transforms as transforms
import numpy as np
from kmeans_pytorch import kmeans


class HOI(nn.Module):
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
        elif opt["arch"] == "resnet18":
            self.extractor = Resnet(opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.K = opt["K"]
        self.embedding_dim = opt["dim"]
        self.normalize = opt["normalize"]
        self.device = device

        self.vq = nn.Parameter(torch.rand(self.K, dim), requires_grad=True)
        self._vq_initialize(self.vq)

        if self.opt["initial"] == None:
            self._vq_initialized_from_batch = True
        else:
            self._vq_initialized_from_batch = False

        self.vq0_update = torch.zeros(self.K, device=self.device)
        # self.dropout = nn.Dropout(p=0.3)

    def _get_temperature(self, cur_iter):
        # TODO get temperature
        # return max((1000 - cur_iter) / 1000, 0.1)  # 0 -> 1, 900 -> 0.1
        return self.opt["temperature"]

    def _get_gate(self, cur_iter):
        # TODO get gate
        return 0.0
        # return max((10000 - cur_iter) / 10000, 0.0)  # 0 -> 1, 1000 -> 0

    def _vq_initialize(self, vq):  # TODO vq initialize
        initial_type = self.opt["initialize"].lower()
        if initial_type == "svd":
            pass
        elif initial_type == "tsp":
            pass
        elif initial_type == "uni":
            nn.init.xavier_uniform_(vq)
        elif initial_type == "normal":
            nn.init.xavier_normal_(vq)
            # nn.init.normal_(vq, std=0.1)
        else:
            raise ValueError(f"Unsupported vq initial type {initial_type}.")

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor, cur_iter) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
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
            p = F.softmax(-distance, dim=1)
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

        return q_feat, assignment, distance

    def _Augmentation(self, x: torch.Tensor):
        # Augmentation = nn.Sequential(
        #     # TODO flip ok?
        #     # Kg.RandomHorizontalFlip(p=0.5),
        #     Kg.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        #     Kg.RandomGrayscale(p=0.2),
        #     # TODO add gaussian blur?
        #     # Kg.RandomGaussianBlur((int(0.1 * x.shape[1]), int(0.1 * x.shape[2])), (0.1, 2.0), p=0.5)
        # )

        Augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            transforms.RandomGrayscale(.2),
            transforms.RandomApply([transforms.GaussianBlur((5, 5))])
        ])

        return Augmentation(x)

    def forward(self, x: torch.Tensor, cur_iter, is_pos: bool = False, local_rank: int = -1):
        # if is_pos:  # model_pos_output
        #     feat, head = self.extractor(x)
        #     return feat, head
        # if self.training:
        #     I1 = self._Augmentation(x)
        #     I2 = self._Augmentation(x)
        #
        #     feat1, x1 = self.extractor(I1)  # (32, 384, 28, 28)
        #     qx1, assignment1, distance1 = self._vector_quantize(x1, self.vq, cur_iter)  # (32, 384, 28, 28)
        #
        #     feat2, x2 = self.extractor(I2)  # (32, 384, 28, 28)
        #     qx2, assignment2, distance2 = self._vector_quantize(x2, self.vq, cur_iter)  # (32, 384, 28, 28)
        #
        #     if cur_iter % 25 == 0:
        #         print("vq", torch.topk(self.vq0_update, 20).values,
        #               torch.topk(self.vq0_update, 20, largest=False).values)
        #     return [x1, x2], [qx1, qx2], [assignment1, assignment2], [distance1, distance2], 0, [feat1, feat2]
        # else:
        #     feat, x = self.extractor(x)
        #     qx, assignment, distance = self._vector_quantize(x, self.vq, cur_iter)
        #
        #     return x, qx, assignment, distance
        # SPQv.
        if self.training:
            I1 = self._Augmentation(x)
            I2 = self._Augmentation(x)

            feat1, x1 = self.extractor(I1)  # (32, 384, 28, 28)
            qx1, assignment1, distance1 = self._vector_quantize(x1, self.vq, cur_iter)  # (32, 384, 28, 28)

            feat2, x2 = self.extractor(I2)  # (32, 384, 28, 28)
            qx2, assignment2, distance2 = self._vector_quantize(x2, self.vq, cur_iter)  # (32, 384, 28, 28)

            if cur_iter % 25 == 0 and local_rank == 0:
                print("vq", torch.topk(self.vq0_update, 20).values,
                      torch.topk(self.vq0_update, 20, largest=False).values)
            return [x1, x2], [qx1, qx2], [assignment1, assignment2], [distance1, distance2], None, [feat1, feat2]
        else:
            feat, x = self.extractor(x)
            qx, assignment, distance = self._vector_quantize(x, self.vq, cur_iter)

            return x, qx, assignment, distance

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
