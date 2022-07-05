import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self,
                 opt: dict
                 ):
        super().__init__()

        self.embedding_dim = opt["embedding_dim"]
        self.K = opt["K"]  # TODO maybe auxiliary extra-cluster?
        self.e_weight = opt["e_weight"]

        self.embedding = nn.Embedding(self.K, self.embedding_dim).cuda()
        # TODO initialize maybe SVD?
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, inputs: torch.Tensor):
        # inputs : (b, 70, 28, 28)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # (b, 28, 28, 70)
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim)  # (b * 28 * 28, 70)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)  # (flat - embed.weight) ^ 2
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))  # (b * 28 * 28, 70)
        # encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (b * 28 * 28, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.K, device=inputs.device)  # (b * 28 * 28, K)
        encodings.scatter_(1, encoding_indices, 1)  # label one-hot vector
        # quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)  # (b, 28, 28, dim)

        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # TODO momentum update
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.e_weight * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()  # TODO argmin gradient
        vq_dict = {"e_vq": e_latent_loss, "q_vq": q_latent_loss}

        return loss, distances.view(input_shape[0], input_shape[1], input_shape[2], self.K).permute(0, 3, 1, 2).contiguous(), vq_dict
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), vq_dict
