import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    vocab_size: int
    dim: int # Synonymous with d_model. Model's dimensionality (seen in embedding dims for example).
    inter_dim: int # Dimension of the fan out in the layers with a dense FFN.
    moe_inter_dim: int # Dimension of the fan out in the MoE layers.
    n_layers: int # Number of layers in the model. (Number of transformer blocks).
    n_dense_layers: int # How many of the first layers of the model use a dense FFN instead of an MoE router with expert FFNs.
    n_heads: int # How many attention heads there are. NOTE: This will get a little tricky due to using Multihead and Multiquery Attention.
    n_routed_experts: int # How many experts there total in the model (sum of all distinct experts for each layer).
    n_shared_experts: int # How many experts appear in all layers and is ALWAYS activated aside from the topk most likely other experts.
    # Sharing a small number of experts (usually 1 even in the largest models) helps stabilize and regularize MoE routing.
    n_activated_experts: int # How many experts are activated per layer.
    n_expert_groups: int # Instead of routing to across all n_routed_experts, experts are separated into n_expert_groups and first the router
    # Gives each expert group a score.
    n_limited_groups: int # Determines how many of the top groups are used and then the router is applied again to calculate the scores of the individual
    # experts within these groups. Finally, the top n_activated_experts are selected and the token is passed through them.
    score_func: str # Deepseek uses sigmoid. Instead of softmax over experts (computationally expensive), sigmoid clamps all probabilities between 0 and 1 individually.
    route_scale: float # Deepeseek uses 2.5 the idea is since the values produced by score_func are between 0 and 1, meaning the topk will come down to floating points,
    # these scores may struggle in lower precision, so they scale them up.
    q_lora_rank: int # Latent dim for q (smaller than model's dim)
    kv_lora_rank: int # Latent dim for k and v (Deepseek actually uses a dim that is 1/3 of q's latent dim (1536 and 512 respectively)).
    qk_nope_head_dim: int # What dimension of the qk head DOES NOT have positional encoding applied to it.
    qk_rope_head_dim: int # What dimension of the qk head DOES receive positional encoding.
    v_head_dim: int # Dimensions of heads in v.
    # Applying Rotary Positional Encoding to only part of each head is a modern technique. Since attention cannot directly understand position, only applying rope partially
    # across each head allows each head to simultaneously learn how tokens attend to each other while factoring in distance and not accounting for distance. This allows for
    # better long-context accuracy.
    dtype: str # Model dtype.
    scale_fmt: str | None # Will be set to none for this example. This is used internally with how they cache quantization scales.
    index_n_heads: int # The number of attention heads the indexer will have when performing the low precision attention to find the topk most relevant tokens.
    index_head_dim: int # Head dim for the indexer.
    index_topk: int # How many of the topk highest scoring tokens' indices will be returned. (See Indexer to understand).

def apply_rotary_embedding(x, freqs_cis, interleaved=True):
    """
    Applies Rope on input x using precomputed frequencies.
    Credit to Deepseek for this function.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform # type: ignore
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)

class ExplainableIndexer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        self.dim = model_args.dim
        self.n_heads = model_args.index_n_heads
        self.head_dim = model_args.index_head_dim
        self.rope_head_dim = model_args.qk_rope_head_dim
        self.index_topk = model_args.index_topk
        self.q_lora_rank = model_args.q_lora_rank

        self.Wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.Wk = nn.Linear(self.dim, self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, Q_latent: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        B, S, _ = x.shape
        Q = self.Wq_up(Q_latent).view(B, S, self.n_heads, self.head_dim) # Q is brought up to the indexer head count and indexer head dimensions
        Q_positional_encoding, Q_no_positional_encoding = torch.split(Q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1) # Split Q again into two chunks where one will be positionally encoded and the other will not.
        Q_positional_encoding = apply_rotary_embedding(Q_positional_encoding, freqs_cis, False) # Not interleaved due to memory layout. Rotary changes are the exact same mathematically whether interleaved or not.
        Q = torch.cat([Q_positional_encoding, Q_no_positional_encoding], dim=-1)

        K = self.k_norm(self.Wk(x)) 
        K_positional_encoding, K_no_positional_encoding = torch.split(K, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        K_positional_encoding = apply_rotary_embedding(K_positional_encoding, freqs_cis, False)
        K = torch.concat([K_positional_encoding, K_no_positional_encoding], dim=-1)
        Q = rotate_activation(Q) # These apply a hadamard transform which is an orthogonal mixing across channels that makes low rank layers stronger.
        K = rotate_activation(K) # Not 100% neccessary, but definitely help performance, especially in large, deep models like LLMs.

class ExplainableDeepseekSparseAttention(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.softmax_scale = self.qk_head_dim **-0.5

        self.Wq_down = nn.Linear(model_args.dim, model_args.q_lora_rank) # Down projects x into the latent dimension.
        self.q_down_norm = nn.RMSNorm(model_args.q_lora_rank) # Scales down inputs
        self.Wq_up = nn.Linear(model_args.q_lora_rank, model_args.n_heads * self.qk_head_dim) # Up projects latent q into the model dimension.

        self.Wk_down = nn.Linear(model_args.dim, model_args.kv_lora_rank) # Down projects into the latent dimension.
        self.Wk_rope = nn.Linear(model_args.dim, model_args.qk_rope_head_dim) # Creates the keys that will have rope applied to them directly from x.
        self.k_norm = nn.RMSNorm(model_args.kv_lora_rank)
        self.Wk_nope = nn.Linear(model_args.kv_lora_rank, model_args.n_heads * model_args.qk_nope_head_dim) # Creates the keys that will not receive positonal encoding from latent K.
        
        self.Wv_down = nn.Linear(model_args.dim, model_args.kv_lora_rank)
        self.v_norm = nn.RMSNorm(model_args.kv_lora_rank)
        self.Wv_up = nn.Linear(model_args.kv_lora_rank, model_args.n_heads * model_args.v_head_dim) # Up projects V
        
        self.Wo = nn.Linear(model_args.n_heads * model_args.v_head_dim, model_args.dim)

        self.indexer = ExplainableIndexer(model_args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        B, S, _ = x.shape # x is [Batch_size, Sequence_length, model_dim]
        Q = self.Wq_up(self.q_down_norm(self.Wq_down(x))) # Q is projected down into the q latent dimension, normalized, and immediately scaled back up.
        Q = Q.view(B, S, self.n_heads, self.qk_head_dim) # Q was [Batch_size, Sequence_length, dim] -> [Batch_size, Sequence_length, n_heads, qk_head_dim]
        Q_no_positional_encoding, Q_positional_encoding = torch.split(Q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # Split into the two tensors along the head dimension.
        Q_positional_encoding = apply_rotary_embedding(Q_positional_encoding, freqs_cis)
        
        K_latent = self.k_norm(self.Wk_down(x)) # K_latent is [Batch_size, Sequence_length, kv_lora_rank]
        K_positional_encoding = self.Wk_rope(x) # K_positional encoding is [Batch_size, Sequence_length, qk_rope_head_dim]
        K_positional_encoding = apply_rotary_embedding(K_positional_encoding.unsqueeze(2), freqs_cis) # K_positional encoding is [Batch_size, Sequence_length, qk_rope_head_dim] and unsqueeze to [Batch_size, Sequence_length, 1, qk_rope_head_dim]

        V_latent = self.v_norm(self.Wv_down(x)) # V_latent is [Batch_size, Sequence_length, kv_lora_rank]

        if mask is not None: # This is during the calculations before outputting the first token (aka prefilling the KV cache in modern inference) and uses MHA.
            Q = torch.cat([Q_no_positional_encoding, Q_positional_encoding], dim=-1) # Q is now [Batch_size, Sequence_length, n_heads * (qk_nope_head_dim + qk_rope_head_dim)].
            K_no_positional_encoding = self.Wk_nope(K_latent).view(B, S, self.n_heads, self.qk_nope_head_dim) # K_no_positional_encoding is [Batch_size, Sequence_length, n_heads, qk_nope_head_dim]
            V = self.Wv_up(V_latent) # V is [Batch_size, Sequence_length, n_heads * v_head_dim]
            # The expand below matches up n_heads and 1 by expanding the 1 head to ACT like n_heads without actual memory or copy overhead.
            K = torch.cat([K_no_positional_encoding, K_positional_encoding.expand(-1, -1, self.n_heads, -1)], dim=-1) # K is now [Batch_size, Sequence_length, n_heads, (qk_nope_head_dim + qk_rope_head_dim) = qk_head_dim].
            scores = torch.einsum("bshd,bthd->bsht", Q, K).mul_(self.softmax_scale) # Computes attention scores between Q and K and scales.

        else:
            Q_no_positional_encoding = torch.einsum("bshd,hdc->bshc", Q_no_positional_encoding, )
            



