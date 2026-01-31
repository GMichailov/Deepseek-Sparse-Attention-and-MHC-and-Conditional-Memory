import torch
import torch.nn as nn
from deepseek_sparse_attention import ModelArgs

class HyperConnection(nn.Module):
    def __init__(self, model_args: ModelArgs, n_streams):
        super().__init__()
        self.n_streams = n_streams
        self.dim = model_args.dim
        self.mix_old = nn.Parameter(torch.eye(n_streams))
        self.mix_new = nn.Parameter(0.01 * torch.randn(n_streams, n_streams))
        self.norm = nn.RMSNorm(model_args.dim)

    def forward(self, h, F):
        # h = [B, S, n_streams, dim]
        # Mapping F = [B, S, n_streams, dim] to [B, S, n_streams, dim]
        h_norm = self.norm(h)
        f_out = F(h_norm)
        h_next = (
            torch.einsum("ij,bsnd->bsnd", self.mix_old, h) + torch.einsum("ij,bsnd->bsnd", self.mix_new, f_out) # mixes previous stream info and adds to new one.
        )
        return h_next
