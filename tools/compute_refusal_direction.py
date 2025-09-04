#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute r_hat for item <item_id> by comparing hidden states at the decision token:
    r_hat = normalize(mu_neg - mu_pos)

- Reuses LLaRA prompt construction (TrainCollater) and wrap_emb().
- Builds two groups of up to N=100 each:
  * pos: next == item_id   (69)
  * neg: next != item_id, but candidates FORCED to include item_id

Saves:
  - refusal_vec.pt: {'r_hat','mu_pos','mu_neg','meta':{...}}
"""
import os, json, ast, random
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# LLaRA imports (from your repo)
from model.model_interface import MInterface
from data.data_interface import TrainCollater
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU

# ---------- Helpers for MovieLens title mapping ----------
def _mv_title_fix(s: str) -> str:
    for sub in [", The", ", A", ", An"]:
        if sub in s:
            return sub[2:] + " " + s.replace(sub, "")
    return s

def load_item_id2name(data_dir: str):
    path = Path(data_dir) / "u.item"
    id2name = {}
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            ll = line.strip("\n").split("|")
            id2name[int(ll[0]) - 1] = _mv_title_fix(ll[1][:-7])
    return id2name

# ---------- DF preparation (mirrors data/movielens_data.py) ----------
def remove_padding(seq_pairs, pad_id: int, pad_rating: int = 0):
    x = list(seq_pairs)
    for _ in range(10):
        try:
            x.remove((pad_id, pad_rating))
        except ValueError:
            break
    return x

def to_ids_list(x):
    # from [(id,r), ...] -> [id,...]
    return [int(t[0]) for t in x]

def load_and_prepare_df(df_path: str, data_dir: str, padding_item_id: int):
    df = pd.read_pickle(df_path)  # columns: seq, len_seq, next (tuple)
    # filter len>=3 to match repo
    df = df[df["len_seq"] >= 3].copy()

    # unpad copies for sampling / titles
    df["seq_unpad_pairs"] = df["seq"].apply(lambda s: remove_padding(s, padding_item_id, 0))
    df["seq_unpad"] = df["seq_unpad_pairs"].apply(to_ids_list)

    # convert originals to id lists (keeps padded length=10)
    df["seq"] = df["seq"].apply(to_ids_list)
    df["next"] = df["next"].apply(lambda t: int(t[0]))

    # titles
    id2name = load_item_id2name(data_dir)
    df["seq_title"] = df["seq_unpad"].apply(lambda ids: [id2name[i] for i in ids])
    df["next_item_name"] = df["next"].apply(lambda i: id2name[i])
    return df, id2name

# ---------- Candidate builders ----------
def force_candidates_neg(seq_unpad, true_next, item_id, cans_num, universe_ids):
    """
    For NEG group: ensure both <true_next> and <item_id> are present.
    Fill the rest with random items not in seq and not equal to either.
    """
    ban = set(seq_unpad) | {true_next, item_id}
    pool = [i for i in universe_ids if i not in ban]
    need = max(cans_num - 2, 0)
    if len(pool) < need:
        pool = (pool * ((need // max(1, len(pool))) + 2))[:need]
    sample = random.sample(pool, need) if len(pool) >= need else pool
    cands = sample + [true_next, item_id]
    random.shuffle(cands)
    return cands

def candidates_pos(seq_unpad, true_next, cans_num, universe_ids):
    """
    For POS group (true_next == item_id): include the true next + (cans_num-1) random negatives.
    """
    ban = set(seq_unpad) | {true_next}
    pool = [i for i in universe_ids if i not in ban]
    need = max(cans_num - 1, 0)
    if len(pool) < need:
        pool = (pool * ((need // max(1, len(pool))) + 2))[:need]
    sample = random.sample(pool, need) if len(pool) >= need else pool
    cands = sample + [true_next]
    random.shuffle(cands)
    return cands

# ---------- Collation helpers ----------
def build_samples(df, id2name, item_id, cans_num, neg=False):
    """
    Create a list[dict] samples shaped like MovielensData.__getitem__ output.
    """
    universe_ids = sorted(id2name.keys())
    samples = []
    for _, row in df.iterrows():
        seq_unpad = row["seq_unpad"]
        seq_title = row["seq_title"]
        next_id = int(row["next"])
        next_name = row["next_item_name"]
        len_seq = int(row["len_seq"])

        if neg:
            cands = force_candidates_neg(seq_unpad, next_id, item_id, cans_num, universe_ids)
        else:
            cands = candidates_pos(seq_unpad, next_id, cans_num, universe_ids)

        cans_name = [id2name[c] for c in cands]

        samples.append({
            "seq": row["seq"],                    # (possibly padded to length 10)
            "seq_name": seq_title,                # list[str], unpadded titles
            "len_seq": len_seq,                   # int
            "cans": cands,                        # list[int]
            "cans_name": cans_name,               # list[str]
            "len_cans": cans_num,                 # int
            "item_id": next_id,                   # int (ground-truth next)
            "item_name": next_name,               # str
            "correct_answer": next_name,          # str
        })
    return samples

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------- Vector math ----------
def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2) + 1e-12)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--rec_model_path", required=True)
    ap.add_argument("--model_name", default="mlp_projector")
    ap.add_argument("--prompt_path", required=True)
    ap.add_argument("--item_id", type=int, default=69)
    ap.add_argument("--cans_num", type=int, default=10)
    ap.add_argument("--padding_item_id", type=int, default=1682)
    ap.add_argument("--n_per_group", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="./output")

    # NEW: how to bring in LLaRA
    ap.add_argument("--ckpt_path", type=str, default=None, help="Path to LLaRA Lightning ckpt (e.g., checkpoints/movielens/last.ckpt)")
    ap.add_argument("--peft_dir", type=str, default=None, help="Path to PEFT adapter dir if you exported LoRA separately")
    ap.add_argument("--llm_tuning", type=str, default="freeze_lora", choices=["lora","freeze","freeze_lora"])

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prep DF
    df, id2name = load_and_prepare_df(args.df_path, args.data_dir, args.padding_item_id)

    pos_pool = df[df["next"] == args.item_id].sample(n=min(args.n_per_group, (df["next"] == args.item_id).sum()),
                                                     random_state=args.seed)
    neg_pool = df[df["next"] != args.item_id].sample(n=min(args.n_per_group, (df["next"] != args.item_id).sum()),
                                                     random_state=args.seed)

    print(f"[INFO] pos={len(pos_pool)} neg={len(neg_pool)} (capped at {args.n_per_group} each)")

    pos_samples = build_samples(pos_pool, id2name, args.item_id, args.cans_num, neg=False)
    neg_samples = build_samples(neg_pool, id2name, args.item_id, args.cans_num, neg=True)

    # 2) Load LLaRA model interface (tokenizer, llama, projector, rec model)
    mi = MInterface(
        llm_path=args.llm_path,
        rec_model_path=args.rec_model_path,
        model_name=args.model_name,
        lr=1e-3, lr_scheduler="cosine", max_epochs=1, batch_size=1,  # dummies to satisfy ctor
        rec_size=64, rec_embed="SASRec",
        weight_decay=1e-5,
        lora_r=8, lora_alpha=32, lora_dropout=0.1,
        llm_tuning=args.llm_tuning,     # <-- IMPORTANT: ensure LoRA modules exist for inference
        peft_dir=args.peft_dir,         # <-- if you have a PEFT adapter directory
        peft_config=None
    )
    # If you trained with Lightning and only have a single .ckpt that includes LoRA+projector:
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        mi.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[INFO] Loaded LLaRA checkpoint: {args.ckpt_path}")

    mi.eval()

    # Put everything on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mi.llama_model.to(device)
    mi.projector.to(device)
    try:
        mi.rec_model.to(device)
    except Exception:
        pass  # if rec model stays on CPU and projector/device move happen inside, fine

    # Load prompts
    with open(args.prompt_path, "r") as f:
        prompt_list = [ln.strip() for ln in f if ln.strip()]
    collater = TrainCollater(prompt_list=prompt_list, llm_tokenizer=mi.llama_tokenizer, train=False)

    def collect_mean(samples):
        feats = []
        for batch_samples in chunk(samples, args.batch_size):
            new_batch = collater(batch_samples)
            # Move tensors to device
            for k in ["seq","cans","len_seq","len_cans","item_id"]:
                if isinstance(new_batch[k], torch.Tensor):
                    new_batch[k] = new_batch[k].to(device)
            for k in ["input_ids","attention_mask"]:
                new_batch["tokens"][k] = new_batch["tokens"][k].to(device)

            # Build inputs_embeds with rec embeddings injected
            input_embeds = mi.wrap_emb(new_batch)

            with torch.no_grad():
                out = mi.llama_model(
                    inputs_embeds=input_embeds,
                    attention_mask=new_batch["tokens"]["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
            hs_last = out.hidden_states[-1]           # [B,S,H]
            attn = new_batch["tokens"]["attention_mask"]  # [B,S]
            idx = attn.sum(dim=1) - 1                 # [B]
            # gather per-sample hidden at decision token
            for i in range(hs_last.size(0)):
                feats.append(hs_last[i, idx[i].item(), :].to(torch.float32).detach())

        H = torch.stack(feats, dim=0)  # [N,H]
        return H.mean(dim=0)           # [H]

    print("[INFO] Collecting μ_pos …")
    mu_pos = collect_mean(pos_samples)
    print("[INFO] Collecting μ_neg …")
    mu_neg = collect_mean(neg_samples)

    r_hat_raw = (mu_neg - mu_pos)
    r_hat = F.normalize(r_hat_raw, dim=0)

    out = {
        "r_hat": r_hat.cpu(),
        "mu_pos": mu_pos.cpu(),
        "mu_neg": mu_neg.cpu(),
        "meta": {
            "df_path": str(args.df_path),
            "data_dir": str(args.data_dir),
            "llm_path": str(args.llm_path),
            "rec_model_path": str(args.rec_model_path),
            "model_name": args.model_name,
            "prompt_path": args.prompt_path,
            "item_id": args.item_id,
            "cans_num": args.cans_num,
            "padding_item_id": args.padding_item_id,
            "n_per_group": args.n_per_group,
            "batch_size": args.batch_size,
        }
    }
    vec_path = out_dir / f"refusal_vec_item{args.item_id}.pt"
    torch.save(out, vec_path)
    print(f"[DONE] Saved vector to: {vec_path.resolve()}")


if __name__ == "__main__":
    main()