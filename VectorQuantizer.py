import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self,
                 opt: dict,
                 embedding_dim: int
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.K = opt["K"]
        self.e_weight = opt["e_weight"]
        self.update_index = torch.zeros(self.K)
        self.manage_weight = opt["manage_weight"]


        self.embedding = nn.Embedding(self.K, self.embedding_dim).cuda()
        # TODO initialize maybe SVD?
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, inputs: torch.Tensor, iteration: int = None, is_diff: bool = False):
        # inputs : (b, 70, 28, 28)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # (b, 28, 28, 70)

        # normalize L2 norm
        inputs = F.normalize(inputs, dim=-1)
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim)  # (b * 28 * 28, 70)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))  # (b * 28 * 28, K)

        # Entropy Loss
        # (25088, 70)
        if self.manage_weight > 0:
            p = F.softmax(distances, dim=1)
            entropy = -p * torch.log(p + 1e-8)
            entropy = torch.sum(entropy, dim=-1)  # (25088,)
            self.intra_loss = entropy.mean()  # minimization

            avg_p = p.mean(0)  # (K,)
            avg_entropy = -avg_p * torch.log(avg_p + 1e-8)
            avg_entropy = torch.sum(avg_entropy, dim=-1)  # (1,)
            self.inter_loss = -avg_entropy  # maximization

        # encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (b * 28 * 28, 1)

        # TODO need to optimize
        for x in encoding_indices:
            self.update_index[x] += 1

        encodings = torch.zeros(encoding_indices.shape[0], self.K, device=inputs.device)  # (b * 28 * 28, K)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector
        # quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)  # (b, 28, 28, dim)

        # loss
        q_latent_loss = F.mse_loss(inputs.detach(), quantized)
        e_latent_loss = F.mse_loss(inputs, quantized.detach())

        if self.manage_weight > 0:
            loss = q_latent_loss + self.e_weight * e_latent_loss + self.manage_weight * (self.intra_loss + self.inter_loss)
            vq_dict = {"e_vq": e_latent_loss, "q_vq": q_latent_loss, "intra": self.intra_loss, "inter": self.inter_loss}
        else:
            loss = q_latent_loss + self.e_weight * e_latent_loss
            vq_dict = {"e_vq": e_latent_loss, "q_vq": q_latent_loss}

        quantized = inputs + (quantized - inputs).detach()  # TODO argmin gradient

        if iteration is not None and iteration % 100 == 0:
            ratio = torch.div(self.update_index, sum(self.update_index))
            print("Memory Bank Status : ", self.update_index, sum(self.update_index))
            print("Ratio              : ", ratio)
            print("Top10              : ", torch.topk(ratio, 5).values)
            print("Bottom10           : ", torch.topk(ratio, 5, largest=False).values)

        if is_diff:
            return loss, distances.view(input_shape[0], input_shape[1], input_shape[2], self.K).permute(0, 3, 1,
                                                                                                        2).contiguous(), vq_dict
        else:
            return loss, quantized.permute(0, 3, 1, 2).contiguous(), vq_dict


# TODO
class MomentumVectorQuantize(nn.Module):
    def __init__(self, opt: dict, embedding_dim: int, eps: float = 1e-5):
        super().__init__()

        self.dim = embedding_dim
        self.n_embed = opt["K"]
        self.e_weight = opt["e_weight"]
        self.momentum_type = opt["is_momentum_type"].lower()
        self.max_decay = opt.get("max_decay_momentum", 0.99)
        self.min_decay = opt.get("min_decay_momentum", 0)
        self.steps = opt.get("steps", 0)
        self.eps = eps

        embed = torch.randn(embedding_dim, self.n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input: torch.Tensor, iteration: int = 0, is_diff: bool = False):
        # inputs : (b, 70, 28, 28)
        input = input.permute(0, 2, 3, 1).contiguous()  # (b, 28, 28, 70)
        input_shape = input.shape

        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            # TODO momentum_decay_scheduling
            if self.momentum_type == "linear":
                self.decay = ((self.max_decay - self.min_decay) / 30530) * iteration + self.min_decay
            elif self.momentum_type == "normal":
                self.decay = self.max_decay
            elif self.momentum_type == "step":
                if iteration <= self.steps:
                    self.decay = ((self.max_decay - self.min_decay) / 1000) * iteration + self.min_decay
                else:
                    self.decay = self.max_decay
            else:
                raise ValueError(f"Unsupported momentum type {self.momentum_type}.")

            if iteration is not None and iteration % 100 == 0:
                print(f"f{iteration}-update weight ={self.decay}")

            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        # return quantize, diff, embed_ind

        # TODO momentum update
        e_latent_loss = F.mse_loss(quantize.detach(), input)
        vq_dict = {"e_vq": e_latent_loss}

        e_latent_loss = self.e_weight * e_latent_loss
        return e_latent_loss, quantize.permute(0, 3, 1, 2).contiguous(), vq_dict

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
