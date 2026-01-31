import torch
import torch.nn as nn
from torch.optim import AdamW
import json
from deepseek_sparse_attention import ModelArgs, LLM
from dataclasses import dataclass
from tqdm import trange
import os
from typing import Dict, Any, Optional, Tuple

class Tokenizer:
    def __init__(self, text_path="dataset.txt"):
        # Using same dataset each time that is static just some quick one char tokenizer.
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.replace("\n", " ")
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])
    
class DatasetManager:
    def __init__(self, seq_len, device="cuda", text_path="dataset.txt"):
        self.seq_len = seq_len
        self.device = device
        self.tokenizer = Tokenizer()
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.replace("\n", " ")
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.num_tokens = len(self.tokens)
        stride = 1
        self.num_sequences = self.num_tokens - (self.seq_len + 1) + 1  # = num_tokens - seq_len - 1
        self.ptr = 0
        print(f"Dataset loaded:")
        print(f"  Total tokens: {self.num_tokens}")
        print(f"  Seq len: {seq_len}")
        print(f"  Total sequences: {self.num_sequences}")

    def get_batch(self, batch_size):
        """
        Returns:
            x: [B, seq_len]
            y: [B, seq_len]
        """
        ix = torch.randint(0, self.num_sequences, (batch_size,))

        xs = []
        ys = []

        for i in ix:
            start = i.item()
            end = start + self.seq_len

            x = self.tokens[start:end]
            y = self.tokens[start+1:end+1]

            xs.append(x)
            ys.append(y)

        x = torch.stack(xs).to(self.device)
        y = torch.stack(ys).to(self.device)

        return x, y


    def reset(self):
        self.ptr = 0

    def remaining_sequences(self):
        return self.num_sequences - self.ptr

def precompute_freqs_cis(
    max_seq_len: int,
    rope_dim: int,
    rope_theta: float = 10000.0,
    device: str = "cuda",
):
    """
    Returns freqs_cis: complex64 tensor of shape [max_seq_len, rope_dim//2]
    Compatible with the common "view_as_complex + multiply" RoPE implementation.
    """
    assert rope_dim % 2 == 0, "rope_dim must be even for RoPE"
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # [T]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))  # [rope_dim/2]
    freqs = torch.outer(t, inv_freq)  # [T, rope_dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def save_checkpoint(path: str, llm: nn.Module, optim: torch.optim.Optimizer, epoch: int, global_step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = llm._orig_mod if hasattr(llm, "_orig_mod") else llm  # type: ignore
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(), #type: ignore
        "optim_state_dict": optim.state_dict(),
    }
    torch.save(ckpt, path)

@dataclass
class TrainingArgs:
    lr: float
    dropout_p: float
    beta1: float
    beta2: float
    weight_decay: float
    fused: bool
    epochs: int
    batch_size: int
    grad_accum_steps: int
    seq_len: int
    save_steps: int

training_args = TrainingArgs(
    lr=3e-4,
    dropout_p=0.1,
    beta1=0.9, beta2=0.95,
    weight_decay=0.1,
    fused=True, epochs=5, batch_size=64, grad_accum_steps=8,
    seq_len=256, save_steps=200
)

dataset_manager = DatasetManager(training_args.seq_len)

model_args = ModelArgs(
    vocab_size = dataset_manager.tokenizer.vocab_size,
    dim=768, inter_dim=2304, moe_inter_dim=1536,
    n_layers=12, n_dense_layers=12, n_heads=12,
    n_routed_experts=0, n_shared_experts=0, n_activated_experts=0, n_expert_groups=0,
    n_limited_groups=0, score_func="softmax", route_scale=2.5,
    q_lora_rank=512, kv_lora_rank=256, qk_nope_head_dim=48, qk_rope_head_dim=32,
    v_head_dim=64, dtype="bfloat16", scale_fmt=None,
    index_n_heads=0, index_head_dim=0, index_topk=0
)

def debug_model_size(model_args):
    model = LLM(model_args, 0.0)
    print(sum(p.numel() for p in model.parameters()))

def train(dataset_manager: DatasetManager, training_args: TrainingArgs, model_args: ModelArgs):
    llm = LLM(model_args, training_args.dropout_p).to("cuda", dtype=torch.bfloat16)
    llm._norms_in_fp32()
    llm.train()

    freqs_cis = precompute_freqs_cis(
        max_seq_len=training_args.seq_len,
        rope_dim=model_args.qk_rope_head_dim,
        rope_theta=getattr(model_args, "rope_theta", 10000.0),
        device="cuda",
    )

    optim = AdamW(
        llm.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        weight_decay=training_args.weight_decay,
        fused=training_args.fused,
    )
    optim.zero_grad(set_to_none=True)

    llm = torch.compile(llm)
    loss_function = nn.CrossEntropyLoss()

    steps_per_epoch = dataset_manager.num_sequences // training_args.batch_size
    micro_bs = training_args.batch_size // training_args.grad_accum_steps
    assert training_args.batch_size % training_args.grad_accum_steps == 0

    global_step = 0
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, training_args.epochs + 1):
        pbar = trange(steps_per_epoch, desc=f"Epoch {epoch}")
        for step in pbar:
            total_loss = 0.0

            for _ in range(training_args.grad_accum_steps):
                tokens_tensor, targets_tensor = dataset_manager.get_batch(micro_bs)
                seqlen = tokens_tensor.size(1)
                freqs = freqs_cis[:seqlen]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = llm(tokens_tensor, freqs)
                    loss = loss_function(
                        logits.reshape(-1, logits.size(-1)),
                        targets_tensor.reshape(-1),
                    )
                    loss = loss / training_args.grad_accum_steps

                loss.backward()
                total_loss += float(loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)  # type: ignore
            optim.step()
            optim.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=f"{total_loss:.4f}", gstep=global_step)

            if training_args.save_steps > 0 and (global_step % training_args.save_steps == 0):
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                save_checkpoint(ckpt_path, llm, optim, epoch, global_step) #type: ignore

def load_llm_from_checkpoint(
    ckpt_path: str,
    model_args: ModelArgs,
    dropout_p: float = 0.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Constructs LLM from model_args, loads checkpoint, returns (model, ckpt_meta).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    llm = LLM(model_args, dropout_p).to(device=device, dtype=dtype)
    llm._norms_in_fp32()  # match your training stability behavior
    llm.load_state_dict(ckpt["model_state_dict"], strict=True)
    llm.eval()

    meta = {
        "epoch": ckpt.get("epoch", None),
        "global_step": ckpt.get("global_step", None),
    }
    return llm, meta


@torch.no_grad()
def run_inference(
    prompt: str,
    ckpt_path: str,
    model_args: ModelArgs,
    tokenizer: Tokenizer,
    seq_len: int = 256,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    rope_theta: float = 10000.0,
) -> Dict[str, Any]:
    """
    If max_new_tokens == 0:
        returns logits for the prompt only.
    Else:
        autoregressively generates max_new_tokens and returns generated text.

    Returns dict with:
      - "text": decoded output (prompt + generated)
      - "tokens": LongTensor [1, T]
      - "logits": (optional) logits for final forward pass
      - "meta": checkpoint metadata
    """
    llm, meta = load_llm_from_checkpoint(
        ckpt_path=ckpt_path,
        model_args=model_args,
        dropout_p=0.0,
        device=device,
        dtype=dtype,
    )

    # Precompute RoPE freqs once up to seq_len (context window)
    freqs_cis = precompute_freqs_cis(
        max_seq_len=seq_len,
        rope_dim=model_args.qk_rope_head_dim,
        rope_theta=rope_theta,
        device=device,
    )

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    def _forward_context(tok: torch.Tensor) -> torch.Tensor:
        # tok: [1, Tctx]
        Tctx = tok.size(1)
        freqs = freqs_cis[:Tctx]
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = llm(tok, freqs)  # [1, Tctx, vocab]
        return logits

    # If you only want logits for the prompt:
    if max_new_tokens <= 0:
        ctx = tokens[:, -seq_len:]
        logits = _forward_context(ctx)
        return {
            "text": tokenizer.decode(tokens[0].tolist()),
            "tokens": tokens,
            "logits": logits,
            "meta": meta,
        }

    # Otherwise: generate
    for _ in range(max_new_tokens):
        ctx = tokens[:, -seq_len:]  # crop to context window
        logits = _forward_context(ctx)
        next_logits = logits[:, -1, :]  # [1, vocab]

        if temperature is None or temperature <= 0.0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1, 1]
        else:
            next_logits = next_logits / float(temperature)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                cutoff = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

        tokens = torch.cat([tokens, next_id], dim=1)

    out_text = tokenizer.decode(tokens[0].tolist())
    return {
        "text": out_text,
        "tokens": tokens,
        "meta": meta,
    }

out = run_inference(
    prompt="Who art thou? ",
    ckpt_path="checkpoints/ckpt_step_800.pt",
    model_args=model_args,
    tokenizer=dataset_manager.tokenizer,
    seq_len=256,
    max_new_tokens=1000,
    temperature=0.8,
    top_k=50,
)

print(out["meta"])
print(out["text"])

#train(dataset_manager, training_args, model_args)
