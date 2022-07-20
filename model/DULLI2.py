from typing import List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.layer_utils import ClusterLookup
from model.dino.DinoFeaturizer import DinoFeaturizer


class DULLI2(nn.Module):
    # opt["model"]
    def __init__(self,
                 opt: dict,
                 n_classes: int
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

        self.num_vq = opt["num_vq"]  # 4
        self.K = opt["K"]  # 1024
        self.embedding_dim = opt["dim"]

        k_list = [self.K, n_classes]  # TODO temp list
        channel_list = [self.embedding_dim, 2048]  # TODO my mind

        assert len(k_list) == len(channel_list) == self.num_vq
        self.cluster_probe = ClusterLookup(channel_list[-1], n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(channel_list[-1], n_classes, (1, 1))

        self.num_heads = self.num_vq - 1
        self.head = self._generate_head(channel_list)

        self.register_buffer("vq0", torch.randn(k_list[0], channel_list[0]))
        self.register_buffer("vq1", torch.randn(k_list[1], channel_list[1]))

    def _get_temperature(self, cur_iter):
        # TODO get temperature
        return 0.01

    def _get_gate(self, cur_iter):
        # TODO get gate
        return 0.0

    def _generate_head(self, channel_list: List):  # TODO maybe stego head style? -> kernel (1,1) maybe (3,3)
        head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=channel_list[i], out_channels=channel_list[i + 1], kernel_size=(3, 3), padding=1,
                          padding_mode="reflect"),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(in_channels=channel_list[i + 1], out_channels=channel_list[i + 1], kernel_size=(1, 1))
            )
            for i in range(self.num_vq - 1)
        ])

        return head

    def _vector_quantize(self, feat: torch.Tensor, codebook: torch.Tensor,
                         cur_iter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param feat:            (batch_size, ch, h, w)
        :param codebook:        (K, ch)
        :return:
                output feat:    (batch_size, ch, h, w)
                assignment:     (batch_size, K, h, w)
        """
        b, c, h, w = feat.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        flat = feat.view(-1, c)  # (bhw, c)
        # flat = F.normalize(flat, dim=1)
        code = codebook  # (K, c)

        # distance = flat.unsqueeze(1) - code.unsqueeze(0)  # (bhw, 1, c) - (1, K, c) = (bhw, K, c)
        # distance = torch.sum(distance ** 2, dim=-1)  # (bhw, K)

        distance = (torch.sum(flat ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                    + torch.sum(code ** 2, dim=1)
                    - 2 * torch.matmul(flat, code.t()))  # (bhw, K)
        # smaller distance == higher probability

        prob = torch.softmax(-distance / self._get_temperature(cur_iter), dim=-1)  # (bhw, K)
        assignment = prob.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, K, h, w)

        if self.training:
            with torch.no_grad():

                discrete_assignment = torch.argmax(assignment, dim=1)  # (b, h, w)

        q_feat = torch.matmul(prob, code)  # (bhw, K) x (K, c) = (bhw, c)
        q_feat = q_feat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        # q_feat = torch.einsum("bkhw,kc->bchw", assignment, code)  # (b, c, h, w)

        feat = feat.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        # g = self._get_gate(cur_iter)
        out_feat = feat + (q_feat - feat).detach()

        return out_feat, assignment

    def forward(self, x: torch.Tensor, cur_iter) -> Tuple[
        List[Any], List[Any], List[Any]]:

        # KH original thought
        x0 = self.extractor(x)  # (32, 384, 28, 28)
        qx0, assignment0 = self._vector_quantize(x0, self.vq[0], cur_iter)  # (32, 384, 28, 28)

        # x1 = self.head[0](qx0)  # (32, 512, 28, 28)
        # qx1, assignment1 = self._vector_quantize(x1, self.vq[1], cur_iter)  # (32, 512, 28, 28)
        #
        # x2 = self.head[1](qx1)  # (32, 1024, 28, 28)
        # qx2, assignment2 = self._vector_quantize(x2, self.vq[2], cur_iter)  # (32, 1024, 28, 28)

        # x3 = self.head[2](qx2)  # (32, 2048, 28, 28)
        # qx3, assignment3 = self._vector_quantize(x3, self.vq[3], cur_iter)  # (32, 2048, 28, 28)

        x3 = self.head[0](qx0)  # (32, 2048, 28, 28)
        qx3, assignment3 = self._vector_quantize(x3, self.vq[1], cur_iter)  # (32, 2048, 28, 28)

        return [x0, x3], [qx0, qx3], [assignment0, assignment3]
        # return [x0, x1, x2, x3], [qx0, qx1, qx2, qx3], [assignment0, assignment1, assignment2, assignment3]

    @classmethod
    def build(cls, opt, n_classes):
        # opt = opt["model"]
        m = cls(
            opt=opt,
            n_classes=n_classes
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count
