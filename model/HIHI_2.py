from typing import List, Tuple, Any, Optional
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from utils.layer_utils import ClusterLookup
from model.dino.DinoFeaturizer import DinoFeaturizer
from model.res.fpn import Resnet
from kmeans_pytorch import kmeans


class HIHI2(nn.Module):
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
            self.extractor = DinoFeaturizer(dim, opt, use_head=False)
        elif opt["arch"] == "resnet18":
            self.extractor = Resnet(opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.K1 = opt["vq"]["stage1"]["K"]
        self.K2 = opt["vq"]["stage2"]["K"]
        self.K3 = opt["vq"]["stage3"]["K"]
        self.embedding_dim = opt["dim"]
        self.normalize = opt["normalize"]
        self.device = device
        self.channel = opt["vq"]["channel"]
        self.n_res_block = opt["vq"]["n_res_block"]
        self.n_res_channel = opt["vq"]["n_res_channel"]
        self.decay = opt["vq"]["decay"]
        self.eps = 1e-5
        self.q_weight = opt["vq"]["q_weight"]
        self.e_weight = opt["vq"]["e_weight"]
        self.is_ema = opt["vq"]["is_ema"]

        if self.is_ema:
            print("HIHI2 currently only supports parameterized ver. Changed!")
            self.is_ema = False

        self.vq1_update = torch.zeros(self.K1, device=self.device)
        self.vq2_update = torch.zeros(self.K2, device=self.device)
        self.vq3_update = torch.zeros(self.K3, device=self.device)

        # TODO encoder architecture
        # self.enc_proj = nn.Sequential(
        #     nn.Conv2d(384, dim // 4, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // 4, dim, kernel_size=1, stride=1)
        # )
        self.enc_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.dec_proj = nn.Conv2d(dim, 384, kernel_size=1, stride=1)

        if self.is_ema:
            # EMA style
            # TODO not initialize
            # ----- stage1 ----- #
            embed = torch.randn(dim, self.K1)
            self.register_buffer("vq1", embed)
            self.register_buffer("vq1_cluster_size", torch.zeros(self.K1))
            self.register_buffer("vq1_avg", embed.clone())

            # ----- stage2 ----- #
            embed = torch.randn(dim, self.K2)
            self.register_buffer("vq2", embed)
            self.register_buffer("vq2_cluster_size", torch.zeros(self.K2))
            self.register_buffer("vq2_avg", embed.clone())

            # ----- stage3 ----- #
            embed = torch.randn(dim, self.K3)
            self.register_buffer("vq3", embed)
            self.register_buffer("vq3_cluster_size", torch.zeros(self.K3))
            self.register_buffer("vq3_avg", embed.clone())

            self.register_buffer("scale0", torch.ones(dim, 1, 1))
            self.register_buffer("bias0", torch.zeros(dim, 1, 1))

            self.register_buffer("scale1", torch.ones(dim, 1, 1))
            self.register_buffer("bias1", torch.zeros(dim, 1, 1))

            self.register_buffer("scale2", torch.ones(dim, 1, 1))
            self.register_buffer("bias2", torch.zeros(dim, 1, 1))

        else:
            # Loss style
            # ----- stage1 ----- #
            self.vq1 = nn.Parameter(torch.rand(self.K1, dim), requires_grad=True)
            self._vq_initialize(self.vq1)

            # ----- stage2 ----- #
            self.vq2 = nn.Parameter(torch.rand(self.K2, dim), requires_grad=True)
            self._vq_initialize(self.vq2)

            # ----- stage3 ----- #
            self.vq3 = nn.Parameter(torch.rand(self.K3, dim), requires_grad=True)
            self._vq_initialize(self.vq3)

            self.scale0 = nn.Parameter(torch.ones(dim, 1, 1))
            self.bias0 = nn.Parameter(torch.zeros(dim, 1, 1))

            self.scale1 = nn.Parameter(torch.ones(dim, 1, 1))
            self.bias1 = nn.Parameter(torch.zeros(dim, 1, 1))

            self.scale2 = nn.Parameter(torch.ones(dim, 1, 1))
            self.bias2 = nn.Parameter(torch.zeros(dim, 1, 1))

        if self.opt["initial"] == None:
            self._vq_initialized_from_batch = True
        else:
            self._vq_initialized_from_batch = False

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

    def _ema_quantize(self, feat: torch.Tensor, K, cur_iter: int = -1):
        if K == self.K1:
            embed = self.vq1
            cluster_size = self.vq1_cluster_size
            embed_avg = self.vq1_avg
            vq_update = self.vq1_update
        elif K == self.K2:
            embed = self.vq2
            cluster_size = self.vq2_cluster_size
            embed_avg = self.vq2_avg
            vq_update = self.vq2_update
        elif K == self.K3:
            embed = self.vq3
            cluster_size = self.vq3_cluster_size
            embed_avg = self.vq3_avg
            vq_update = self.vq3_update
        else:
            raise ValueError(f"No required K : {K}")

        b, c, h, w = feat.shape
        k, _ = embed.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        flat = feat.view(-1, c)  # (bhw, c)

        distance = (torch.sum(flat ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                    + torch.sum(embed ** 2, dim=0)
                    - 2 * torch.matmul(flat, embed))  # (bhw, K)

        # smaller distance == higher probability
        _, embed_ind = (-distance).max(1)
        embed_onehot = F.one_hot(embed_ind, K).type(flat.dtype)
        embed_ind = embed_ind.view(*feat.shape[:-1])

        quantize = self.embed_code(embed_ind, embed)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flat.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = cluster_size.sum()
            cluster_size_ = (
                    (cluster_size + self.eps) / (n + K * self.eps) * n
            )
            embed_normalized = embed_avg / cluster_size_.unsqueeze(0)
            embed.data.copy_(embed_normalized)

            vq_update += embed_onehot_sum
            # TODO restart
            if cur_iter % 25 == 0 and K == self.K1:
                top2 = torch.topk(distance, 2).values
                top1_2_distance = top2[:, 0] - top2[:, 1]
                print("top1-2 : ", top1_2_distance.mean())

        diff = self.q_weight * F.mse_loss(quantize.detach(), feat)
        q_feat = feat + (quantize - feat).detach()  # (b, h, w, c)
        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        return q_feat, diff

    def embed_code(self, embed_id, embed):
        return F.embedding(embed_id, embed.transpose(0, 1))

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor, K: int, vq_update, cur_iter: int = -1) -> \
            Tuple[torch.Tensor, torch.Tensor]:
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
        encodings = torch.zeros(distance.shape[0], K, device=flat.device)  # (bhw, K)
        encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)  # (bhw, 1)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector

        # if self.opt["is_weight_sum"]:
        #     p = F.softmax(-distance, dim=1)
        #     q_feat = torch.matmul(p, code)  # (bhw, K) x (K, c) = (bhw, c)
        #     q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)
        q_feat = torch.matmul(encodings, code)
        q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

        q_loss = F.mse_loss(feat.detach(), q_feat)
        e_loss = F.mse_loss(feat, q_feat.detach())

        diff = (self.q_weight * q_loss + self.e_weight * e_loss)
        q_feat = feat + (q_feat - feat).detach()  # (b, h, w, c)

        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        # prob = torch.softmax(-distance / self._get_temperature(cur_iter), dim=1)  # (bhw, K)
        # assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        # calculate update
        if self.training:
            cur_update = torch.sum(encodings, dim=0)
            vq_update += cur_update
            # TODO restart
            if cur_iter % 25 == 0 and K == self.K1:
                top2 = torch.topk(distance, 2).values
                top1_2_distance = top2[:, 0] - top2[:, 1]
                print("top1-2 : ", top1_2_distance.mean())

        return q_feat, diff

    def forward(self, x: torch.Tensor, cur_iter: int = -1, is_pos: bool = False, local_rank: int = -1):
        feat, head = self.extractor(x)  # (b, 384, 28, 28), (b, 384, 28, 28)

        head = self.enc_proj(feat)  # (b, 384, 28, 28)

        if self.is_ema:
            # EMA style
            if not self.training:
                result = self._ema_quantize(head, self.K1)
                return result

            x1, loss1 = self._ema_quantize(head, self.K1, cur_iter)  # (b, 384, 28, 28)
            remain1 = head - x1

            x2, loss2 = self._ema_quantize(remain1, self.K2, cur_iter)  # (b, 384, 28, 28)
            remain2 = remain1 - x2

            x3, loss3 = self._ema_quantize(remain2, self.K3, cur_iter)  # (b, 384, 28, 28)
            # remain3 = remain2 - x3

            raise NotImplementedError("Scale and bias update not yet applied.")
        else:
            # Loss style
            if not self.training:
                result = self._vector_quantize(head, self.vq1, self.K1, self.vq1_update)
                return result

            # Loss style
            head_normalized = (head - self.bias0) / self.scale0
            x1, loss1 = self._vector_quantize(head_normalized, self.vq1, self.K1, self.vq1_update,
                                              cur_iter)  # (b, 384, 28, 28)
            remain1 = head_normalized - x1

            remain1_normalized = (remain1 - self.bias1) / self.scale1
            x2, loss2 = self._vector_quantize(remain1_normalized, self.vq2, self.K2,
                                              self.vq2_update)  # (b, 384, 28, 28)
            remain2 = remain1_normalized - x2

            remain2_normalized = (remain2 - self.bias2) / self.scale2
            x3, loss3 = self._vector_quantize(remain2_normalized, self.vq3, self.K3, self.vq3_update)
            # remain3 = remain2 - x3

            # ----- roll back ---- #
            recon2 = x3 * self.scale2 + self.bias2
            recon1 = (recon2 + x2) * self.scale1 + self.bias1
            recon0 = (recon1 + x1) * self.scale0 + self.bias0

        recon = self.dec_proj(recon0)

        # print status of vq
        if self.training and cur_iter % 25 == 0:
            print(f"[stage1] coarse\ntop : ", torch.topk(self.vq1_update, 20).values,
                  "\nbottom : ", torch.topk(self.vq1_update, 20, largest=False).values)
            print(f"[stage2] middle\ntop : ", torch.topk(self.vq2_update, 20).values,
                  "\nbottom : ", torch.topk(self.vq2_update, 20, largest=False).values)
            print(f"[stage3] fine\ntop : ", torch.topk(self.vq3_update, 20).values,
                  "\nbottom : ", torch.topk(self.vq3_update, 20, largest=False).values)

        return [x1, x2, x3], \
               [loss1, loss2, loss3], \
               recon, feat, head

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
