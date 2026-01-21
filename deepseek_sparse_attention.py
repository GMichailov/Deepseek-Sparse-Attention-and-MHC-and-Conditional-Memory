import torch
import torch.nn as nn

class Indexer(nn.Module):
    def __init__(self, d_model, d_latent, top_k) -> None:
        super().__init__()
        self.Wq_down = nn.Linear(d_model, d_latent, dtype=torch.float8_e4m3fn)
        self.top_k = top_k

    def forward(self, Q, K_down, V_down):
        """
        Q = [batch, num_heads, seq_len, d_model], K_down = [batch, 1 (MQA), seq_len, d_latent]
        """
        last_Q_row = Q[:, :, -1, :].to(torch.float8_e4m3fn)
        last_Q_row = last_Q_row * self.Wq_down
        fuzzy_scores = nn.ReLU(last_Q_row * K_down.transpose(-2, -1))
        # Get top_k from fuzzy scores.
        # Returns K_top_k, V_top_k

        

class DeepseekSparseAttention(nn.Module):
    def __init__(self, num_heads, d_head, d_latent, top_k):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_latent = d_latent        
        d_model = num_heads * d_head
        self.Wq = nn.Linear(d_model, d_model)
        self.Wkv_down = nn.Linear(2 * 1 * d_head, d_latent) # One signifies the num_kv_heads since they used MQA.
        self.Wkv_up = nn.Linear(2 * 1 * d_head, d_model)
        self.Wout = nn.Linear(d_model, d_model)
        self.indexer = Indexer(d_model, d_latent, top_k)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = x @ self.Wq
        Q = Q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1,2)
        KV_down = x @ self.Wkv_down
        KV_down = KV_down.view(batch, seq_len, 2, self.d_head).tranpose(1, 2)
        K_down, V_down = KV_down.chunk(2, dim=1) # Along num_heads
        K_top_down, V_top_down = self.indexer(Q, K_down)
        

