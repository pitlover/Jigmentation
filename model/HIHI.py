from typing import List, Tuple, Any, Optional
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from utils.layer_utils import ClusterLookup
from model.dino.DinoFeaturizer import DinoFeaturizer
from model.res.fpn import Resnet
from kmeans_pytorch import kmeans


class HIHI(nn.Module):
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

        self.K1 = opt["vq"]["stage1"]["K"]
        self.K2 = opt["vq"]["stage2"]["K"]
        self.K3 = opt["vq"]["stage3"]["K"]
        self.embedding_dim = opt["dim"]
        self.normalize = opt["normalize"]
        self.device = device
        self.channel = opt["vq"]["channel"]
        self.n_res_block = opt["vq"]["n_res_block"]
        self.n_res_channel = opt["vq"]["n_res_channel"]

        # ----- stage1 ----- #
        self.vq1 = nn.Parameter(torch.rand(self.K1, dim), requires_grad=True)
        self._vq_initialize(self.vq1)
        self.vq1_update = torch.zeros(self.K1, device=self.device)

        # ----- stage2 ----- #
        self.vq2 = nn.Parameter(torch.rand(self.K2, dim), requires_grad=True)
        self._vq_initialize(self.vq2)
        self.vq2_update = torch.zeros(self.K2, device=self.device)

        # ----- stage3 ----- #
        self.vq3 = nn.Parameter(torch.rand(self.K3, dim), requires_grad=True)
        self._vq_initialize(self.vq3)
        self.vq3_update = torch.zeros(self.K3, device=self.device)

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

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor, K: int, vq_update) -> Tuple[
        torch.Tensor, torch.Tensor]:
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

        diff = (q_loss + e_loss)
        q_feat = feat + (q_feat - feat).detach()  # (b, h, w, c)

        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        # prob = torch.softmax(-distance / self._get_temperature(cur_iter), dim=1)  # (bhw, K)
        # assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        # calculate update
        if self.training:
            cur_update = torch.sum(encodings, dim=0)
            vq_update += cur_update
            # TODO restart

        return q_feat, diff

    # TODO model forward
    def forward(self, x: torch.Tensor, cur_iter, is_pos: bool = False, local_rank: int = -1):
        feat, head = self.extractor(x)  # (b, 384, 28, 28), (b, 384, 28, 28)

        if not self.training:
            result = self._vector_quantize(head, self.vq1, self.K1, self.vq1_update)
            return result

        x1, loss1 = self._vector_quantize(head, self.vq1, self.K1, self.vq1_update)  # (b, 384, 28, 28)

        remain1 = head - x1
        x2, loss2 = self._vector_quantize(remain1, self.vq2, self.K2, self.vq2_update)  # (b, 384, 28, 28)

        remain2 = remain1 - x2
        x3, loss3 = self._vector_quantize(remain2, self.vq3, self.K3, self.vq3_update)

        recon = x1 + x2 + x3

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


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(out_channel, eps=1e-6)

    def forward(self, x: torch.Tensor, is_final: bool = False):
        x = self.blocks(x)  # (b, 64, 14, 14)
        if is_final:
            x = self.norm(x.permute(0, 2, 3, 1))  # (b, h, w, c)
            x = x.permute(0, 3, 1, 2)  # (b, c, h, w)
        return x
