#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_weight_ablation.py

Make a NEW, patched copy of a LLaMA model where writers to the residual stream
are orthogonalized to a given direction r_hat (weight-space directional ablation):

  W' = W - beta * r_hat (r_hat^T W)
  b' = b - beta * r_hat (r_hat^T b)

Patched modules:
  - token embedding: embed_tokens.weight
  - each decoder layer's attention out proj: self_attn.o_proj
  - each decoder layer's MLP out proj:      mlp.down_proj

Inputs:
  --llm_path   : base LLaMA repo/path
  --vec_path   : path to vector file (.pt) with payload['r_hat'] (1D, [hidden_size])
  --beta       : ablation strength (default 1.0)
  --save_dir   : where to save the patched HF model
  --peft_dir   : (optional) PEFT/LoRA adapter dir to attach before patching
  --merge_lora : (flag) if set and a PEFT adapter is attached, merge into base before patching

Usage example:
  python -m tools.apply_weight_ablation \
    --llm_path /path/to/meta-llama/Llama-2-7b-hf \
    --vec_path ./output/refusal_vec_item69.pt \
    --beta 1.0 \
    --save_dir ./llama_patched_item69_beta1

If you have a LoRA adapter to include:
  python -m tools.apply_weight_ablation \
    --llm_path /path/to/meta-llama/Llama-2-7b-hf \
    --peft_dir ./checkpoints/llara_adapter \
    --merge_lora \
    --vec_path ./output/refusal_vec_item69.pt \
    --beta 1.0 \
    --save_dir ./llama_patched_item69_beta1
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# --------- math helpers ---------

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2) + 1e-12)


@torch.no_grad()
def orth_linear_out_(linear: nn.Linear, r_hat: torch.Tensor, beta: float):
    """
    In-place: W' = W - beta * r_hat (r_hat^T W); same for bias if present.
    Shapes:
      W: [out_features (=hidden_size), in_features]
      r_hat: [hidden_size]
    """
    W = linear.weight.data
    r = r_hat.to(W.dtype).to(W.device).view(-1, 1)     # [H,1]
    # W -= beta * r (r^T W)
    W -= beta * (r @ (r.transpose(0, 1) @ W))
    linear.weight.data.copy_(W)

    if linear.bias is not None:
        b = linear.bias.data.view(-1, 1)               # [H,1]
        b -= beta * (r @ (r.transpose(0, 1) @ b))
        linear.bias.data.copy_(b.view(-1))


@torch.no_grad()
def orth_embed_out_(embedding: nn.Embedding, r_hat: torch.Tensor, beta: float):
    """
    In-place on embedding matrix E: E' = E - beta * E (r_hat r_hat^T)
    Shapes:
      E: [vocab_size, hidden_size]
      r_hat: [hidden_size]
    """
    E = embedding.weight.data
    r = r_hat.to(E.dtype).to(E.device).view(-1, 1)     # [H,1]
    P = r @ r.transpose(0, 1)                          # [H,H]
    E -= beta * (E @ P)
    embedding.weight.data.copy_(E)


# --------- model traversal helpers ---------

@torch.no_grad()
def _get_llama_core(hf_model):
    """
    Return (core, layers, embed_tokens) for both plain and PEFT-wrapped LLaMA.
    `core` is a LlamaModel (has .layers and .embed_tokens).
    This handles common HF structure differences across versions/wrappers.
    """
    core = None

    # Most common: plain LlamaForCausalLM exposes .model (a LlamaModel)
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        core = hf_model.model

    # Some versions expose .base_model as the LlamaModel directly
    elif hasattr(hf_model, "base_model") and hasattr(hf_model.base_model, "layers"):
        core = hf_model.base_model

    # PEFT-wrapped models sometimes nest under base_model.model
    elif hasattr(hf_model, "base_model") and hasattr(hf_model.base_model, "model") \
            and hasattr(hf_model.base_model.model, "layers"):
        core = hf_model.base_model.model

    # Fallback: if the object itself is a LlamaModel
    elif hasattr(hf_model, "layers") and hasattr(hf_model, "embed_tokens"):
        core = hf_model

    if core is None:
        raise AttributeError("Could not locate LLaMA core model (layers/embed_tokens).")

    layers = core.layers
    embed = core.embed_tokens
    return core, layers, embed


@torch.no_grad()
def apply_weight_space_ablation_(hf_model, r_hat: torch.Tensor, beta: float = 1.0):
    """
    Apply weight-space directional ablation to the embeddings and each decoder layer writer.
    """
    _, layers, embed = _get_llama_core(hf_model)

    # Token embeddings
    try:
        orth_embed_out_(embed, r_hat, beta)
    except Exception:
        pass

    # Decoder writers (attention out proj, MLP down proj)
    for layer in layers:
        orth_linear_out_(layer.self_attn.o_proj, r_hat, beta)
        orth_linear_out_(layer.mlp.down_proj,   r_hat, beta)


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_path", required=True, help="Base LLaMA path or HF repo id")
    ap.add_argument("--vec_path", required=True, help="Path to vector .pt (payload['r_hat'] 1D)")
    ap.add_argument("--beta", type=float, default=1.0, help="Ablation strength (1.0 = full)")
    ap.add_argument("--save_dir", required=True, help="Directory to save the patched HF model")
    ap.add_argument("--peft_dir", type=str, default=None, help="(Optional) PEFT/LoRA adapter dir to attach")
    ap.add_argument("--merge_lora", action="store_true", help="Merge adapter into base before patching")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load refusal vector
    payload = torch.load(args.vec_path, map_location="cpu")
    if "r_hat" not in payload:
        raise KeyError(f"'r_hat' not found in vector file: {args.vec_path}")
    r_hat = payload["r_hat"]
    if r_hat.ndim != 1:
        raise ValueError(f"Expected 1D r_hat; got shape {tuple(r_hat.shape)}")
    r_hat = _normalize(r_hat.contiguous())

    # Load base model
    print(f"[INFO] Loading base model from: {args.llm_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Optionally attach PEFT adapter dir
    if args.peft_dir:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed but --peft_dir was provided. pip install peft")
        print(f"[INFO] Attaching PEFT adapter from: {args.peft_dir}")
        model = PeftModel.from_pretrained(model, args.peft_dir)

    # Optionally merge LoRA into base before patching
    if args.merge_lora and PEFT_AVAILABLE and isinstance(model, PeftModel):
        print("[INFO] Merging LoRA adapters into base weights")
        model = model.merge_and_unload()

    # Align r_hat dtype/device to model params
    ref_param = next(model.parameters())
    r_hat = r_hat.to(dtype=ref_param.dtype, device=ref_param.device)

    # Apply weight-space ablation
    print(f"[INFO] Applying weight-space ablation with beta={args.beta}")
    apply_weight_space_ablation_(model, r_hat, beta=args.beta)

    # Save patched model
    print(f"[INFO] Saving patched model to: {save_dir.resolve()}")
    model.save_pretrained(save_dir)

    # Save tokenizer alongside (helps loading without extra flags)
    try:
        tok = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
        tok.save_pretrained(save_dir)
    except Exception as e:
        print(f"[WARN] Could not save tokenizer: {e}")

    print("[DONE] Weight-space ablation complete.")

if __name__ == "__main__":
    main()
