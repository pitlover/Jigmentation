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


class HIER(nn.Module):
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

        self.K1 = opt["vq"]["semantic"]["K1"]
        self.K2 = opt["vq"]["class"]["K2"]
        self.embedding_dim = opt["dim"]
        self.normalize = opt["normalize"]
        self.device = device
        self.channel = opt["vq"]["channel"]
        self.n_res_block = opt["vq"]["n_res_block"]
        self.n_res_channel = opt["vq"]["n_res_channel"]

        # ----- semantic ----- #
        self.vq1 = nn.Parameter(torch.rand(self.K1, dim), requires_grad=True)
        self._vq_initialize(self.vq1)
        self.vq1_update = torch.zeros(self.K1, device=self.device)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.encoder_semantic = Encoder(
            in_channel=dim,
            channel=dim,
            n_res_channel=self.n_res_channel,
            n_res_block=self.n_res_block,
            stride=2
        )
        self.conv_semantic = nn.Conv2d(dim, self.channel, 1)

        # ----- class ----- #
        self.vq2 = nn.Parameter(torch.rand(self.K2, dim), requires_grad=True)
        self._vq_initialize(self.vq2)
        self.vq2_update = torch.zeros(self.K2, device=self.device)
        self.encoder_class = Encoder(
            in_channel=dim,
            channel=dim,
            n_res_block=self.n_res_block,
            n_res_channel=self.n_res_channel,
            stride=2)
        self.conv_class = nn.Conv2d(dim + dim, dim, 1)
        self.upsample = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
        self.decoder_class = Decoder(dim, 384, self.channel, self.n_res_block, self.n_res_channel, stride=2)

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

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor, K: int, vq_update, cur_iter) -> Tuple[
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
        encodings = torch.zeros(distance.shape[0], K, device=flat.device)  # (bhw, K)
        encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)  # (bhw, 1)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector

        # if self.opt["is_weight_sum"]:
        #     p = F.softmax(-distance, dim=1)
        #     q_feat = torch.matmul(p, code)  # (bhw, K) x (K, c) = (bhw, c)
        #     q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

        if K == self.K1:
            p = F.softmax(-distance, dim=1)
            q_feat = torch.matmul(p, code)  # (bhw, K) x (K, c) = (bhw, c)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

        elif K == self.K2:  # TODO top1
            q_feat = torch.matmul(encodings, code)
            q_feat = q_feat.view(b, h, w, -1)  # (b, h, w, c)

            q_loss = F.mse_loss(feat.detach(), q_feat)
            e_loss = F.mse_loss(feat, q_feat.detach())

            self.diff = (q_loss + e_loss)
            q_feat = feat + (q_feat - feat).detach()  # (b, h, w, c)
        else:
            raise ValueError("No such vq.")
        q_feat = q_feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        prob = torch.softmax(-distance / self._get_temperature(cur_iter), dim=1)  # (bhw, K)
        assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        # calculate update
        if self.training:
            cur_update = torch.sum(encodings, dim=0)
            vq_update += cur_update
            # TODO restart

        return q_feat, assignment, distance

    # TODO model forward
    def forward(self, x: torch.Tensor, cur_iter, is_pos: bool = False, local_rank: int = -1):
        x_shape = x.shape  # (b, 384, 28, 28), (b, 70, 28, 28)
        feat, head = self.extractor(x)

        if not self.training:
            result = self._vector_quantize(head, self.vq2, self.K2, self.vq2_update, cur_iter)
            return result
        res_semantic = head  # (b, 70, 28, 28)
        res_class = self.encoder_class(res_semantic)

        # TODO conv need after head + before quantized?
        # res_semantic = self.conv_semantic(res_semantic)
        quantized_semantic, assignment_semantic, distance_semantic = self._vector_quantize(res_semantic, self.vq1,
                                                                                           self.K1, self.vq1_update,
                                                                                           cur_iter)  # (b, 70, 28, 28), (b, k1, 28, 28), (-1, k1)
        # need to match shape q_sematic & q_class
        # TODO avgpool or encoder?
        # quantized_semantic = self.avg_pool(quantized_semantic)                # (b, 70, 14, 14)
        conv_quantized_semantic = self.encoder_semantic(quantized_semantic)  # (b, 70, 14, 14)

        res_class = torch.cat([res_class, conv_quantized_semantic], dim=1)  # (b, 70+70, 14, 14)
        res_class = self.conv_class(res_class)  # (b, 70, 14, 14)
        # TODO class_vq -> top1
        quantized_class, assignment_class, distance_class = self._vector_quantize(res_class, self.vq2, self.K2,
                                                                                  self.vq2_update, cur_iter)
        # (b, 70, 14, 14)
        recon = self.decoder_class(quantized_class, is_final=True)
        # upsample class_semantic -> decode (q_semantic, q_class) -> recon
        # upsample_class = self.upsample(quantized_class)  # (b, 70, 28, 28)
        # quant = torch.cat([quantized_semantic, upsample_class], dim=1)
        # dec = self.decoder_class(quant, is_final=True)

        # print status of vq
        if self.training and cur_iter % 25 == 0:
            print(f"*** [{cur_iter}] vq_semantic ***\ntop : ", torch.topk(self.vq1_update, 20).values, "\nbottom : ",
                  torch.topk(self.vq1_update, 20, largest=False).values)
            print(f"*** [{cur_iter}] vq_class ***\ntop : ", torch.topk(self.vq2_update, 20).values, "\nbottom : ",
                  torch.topk(self.vq2_update, 20, largest=False).values)

        return [res_semantic, quantized_semantic, assignment_semantic, distance_semantic], \
               [res_class, quantized_class, assignment_class, distance_class], \
               recon, feat, head, self.diff

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
