#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute a refusal-style direction per LLaMA layer for a target item_id.
For each layer ell:
  r_hat[ell] = normalize(mu_neg[ell] - mu_pos[ell])
where mu_* are mean hidden states at the decision token ("Answer:") at that layer.

Saves: output/refusal_layerwise_item{item_id}.pt
  {
    'r_hat': FloatTensor [num_layers, hidden_size],
    'mu_pos': FloatTensor [num_layers, hidden_size],
    'mu_neg': FloatTensor [num_layers, hidden_size],
    'meta': {...}
  }
"""

import os, sys, json, random
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# --- make repo imports work when run as a script ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# LLaRA imports
from model.model_interface import MInterface
from data.data_interface import TrainCollater

# For unpickling the Rec model (classic fix)
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
import __main__ as _main
_main.SASRec = SASRec
_main.Caser  = Caser
_main.GRU    = GRU

# ---------- MovieLens helpers (mirror of data/movielens_data.py) ----------
def _mv_title_fix(s: str) -> str:
    for sub in [", The", ", A", ", An"]:
        if sub in s:
            return sub[2:] + " " + s.replace(sub, "")
    return s

def load_item_id2name(data_dir: str):
    id2name = {}
    with open(Path(data_dir)/"u.item", "r", encoding="ISO-8859-1") as f:
        for line in f:
            ll = line.strip("\n").split("|")
            id2name[int(ll[0]) - 1] = _mv_title_fix(ll[1][:-7])
    return id2name

def remove_padding(seq_pairs, pad_id: int, pad_rating: int = 0):
    x = list(seq_pairs)
    for _ in range(10):
        try:
            x.remove((pad_id, pad_rating))
        except ValueError:
            break
    return x

def to_ids_list(x):
    return [int(t[0]) for t in x]

def load_and_prepare_df(df_path: str, data_dir: str, padding_item_id: int):
    df = pd.read_pickle(df_path)
    df = df[df["len_seq"] >= 3].copy()
    df["seq_unpad_pairs"] = df["seq"].apply(lambda s: remove_padding(s, padding_item_id, 0))
    df["seq_unpad"] = df["seq_unpad_pairs"].apply(to_ids_list)
    df["seq"]  = df["seq"].apply(to_ids_list)
    df["next"] = df["next"].apply(lambda t: int(t[0]))
    id2name = load_item_id2name(data_dir)
    df["seq_title"] = df["seq_unpad"].apply(lambda ids: [id2name[i] for i in ids])
    df["next_item_name"] = df["next"].apply(lambda i: id2name[i])
    return df, id2name

# ---------- Candidate builders ----------
def force_candidates_neg(seq_unpad, true_next, item_id, cans_num, universe_ids):
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
    ban = set(seq_unpad) | {true_next}
    pool = [i for i in universe_ids if i not in ban]
    need = max(cans_num - 1, 0)
    if len(pool) < need:
        pool = (pool * ((need // max(1, len(pool))) + 2))[:need]
    sample = random.sample(pool, need) if len(pool) >= need else pool
    cands = sample + [true_next]
    random.shuffle(cands)
    return cands

def build_samples(df, id2name, item_id, cans_num, neg=False):
    universe_ids = sorted(id2name.keys())
    samples = []
    for _, row in df.iterrows():
        seq_unpad = row["seq_unpad"]
        seq_title = row["seq_title"]
        next_id   = int(row["next"])
        next_name = row["next_item_name"]
        len_seq   = int(row["len_seq"])
        cands = force_candidates_neg(seq_unpad, next_id, item_id, cans_num, universe_ids) if neg \
                else candidates_pos(seq_unpad, next_id, cans_num, universe_ids)
        cans_name = [id2name[c] for c in cands]
        samples.append({
            "seq": row["seq"],
            "seq_name": seq_title,
            "len_seq": len_seq,
            "cans": cands,
            "cans_name": cans_name,
            "cans_str": ", ".join(cans_name),
            "len_cans": cans_num,
            "item_id": next_id,
            "item_name": next_name,
            "correct_answer": next_name,
        })
    return samples

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2) + 1e-12)

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
    # LLaRA adapters
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--peft_dir", type=str, default=None)
    ap.add_argument("--llm_tuning", type=str, default="freeze_lora", choices=["lora","freeze","freeze_lora"])
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    df, id2name = load_and_prepare_df(args.df_path, args.data_dir, args.padding_item_id)
    pos_pool = df[df["next"] == args.item_id].sample(n=min(args.n_per_group, (df["next"] == args.item_id).sum()),
                                                     random_state=args.seed)
    neg_pool = df[df["next"] != args.item_id].sample(n=min(args.n_per_group, (df["next"] != args.item_id).sum()),
                                                     random_state=args.seed)
    print(f"[INFO] pos={len(pos_pool)} neg={len(neg_pool)} (capped at {args.n_per_group} each)")

    pos_samples = build_samples(pos_pool, id2name, args.item_id, args.cans_num, neg=False)
    neg_samples = build_samples(neg_pool, id2name, args.item_id, args.cans_num, neg=True)

    # 2) Model (with adapters)
    mi = MInterface(
        llm_path=args.llm_path,
        rec_model_path=args.rec_model_path,
        model_name=args.model_name,
        lr=1e-3, lr_scheduler="cosine", max_epochs=1, batch_size=1,
        rec_size=64, rec_embed="SASRec",
        weight_decay=1e-5, lora_r=8, lora_alpha=32, lora_dropout=0.1,
        llm_tuning=args.llm_tuning, peft_dir=args.peft_dir, peft_config=None
    )
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        mi.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[INFO] Loaded LLaRA ckpt: {args.ckpt_path}")
    mi.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mi.llama_model.to(device); mi.projector.to(device)
    try: mi.rec_model.to(device)
    except: pass

    with open(args.prompt_path, "r") as f:
        prompt_list = [ln.strip() for ln in f if ln.strip()]
    collater = TrainCollater(prompt_list=prompt_list, llm_tokenizer=mi.llama_tokenizer, train=False)

    num_layers = mi.llama_model.config.num_hidden_layers
    hidden_size = mi.llama_model.config.hidden_size

    def collect_layer_means(samples):
        # returns [num_layers, hidden_size]
        sums = torch.zeros(num_layers, hidden_size, dtype=torch.float32, device=device)
        count = 0
        for batch in chunk(samples, args.batch_size):
            new_batch = collater(batch)
            for k in ["seq","cans","len_seq","len_cans","item_id"]:
                if isinstance(new_batch[k], torch.Tensor):
                    new_batch[k] = new_batch[k].to(device)
            for k in ["input_ids","attention_mask"]:
                new_batch["tokens"][k] = new_batch["tokens"][k].to(device)

            input_embeds = mi.wrap_emb(new_batch)
            with torch.no_grad():
                out = mi.llama_model(
                    inputs_embeds=input_embeds,
                    attention_mask=new_batch["tokens"]["attention_mask"],
                    output_hidden_states=True, return_dict=True, use_cache=False
                )
            hs = out.hidden_states  # tuple length = num_layers+1 (0 is embeddings)
            attn = new_batch["tokens"]["attention_mask"]  # [B,S]
            last_idx = attn.sum(dim=1) - 1  # [B]
            B = attn.size(0)
            # accumulate per layer
            for ell in range(1, num_layers+1):
                H = hs[ell]  # [B,S,H]
                for i in range(B):
                    sums[ell-1] += H[i, last_idx[i].item(), :].to(torch.float32)
            count += B
        return sums / max(count, 1)

    print("[INFO] Collecting μ_pos (layerwise)…")
    mu_pos = collect_layer_means(pos_samples)
    print("[INFO] Collecting μ_neg (layerwise)…")
    mu_neg = collect_layer_means(neg_samples)

    r_hat = F.normalize(mu_neg - mu_pos, dim=1)  # [L,H]

    payload = {
        "r_hat": r_hat.detach().cpu(),
        "mu_pos": mu_pos.detach().cpu(),
        "mu_neg": mu_neg.detach().cpu(),
        "meta": {
            "item_id": args.item_id,
            "df_path": args.df_path,
            "data_dir": args.data_dir,
            "llm_path": args.llm_path,
            "rec_model_path": args.rec_model_path,
            "model_name": args.model_name,
            "prompt_path": args.prompt_path,
            "cans_num": args.cans_num,
            "padding_item_id": args.padding_item_id,
            "n_per_group": args.n_per_group,
            "batch_size": args.batch_size,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        }
    }
    out_path = Path(args.out_dir) / f"refusal_layerwise_item{args.item_id}.pt"
    torch.save(payload, out_path)
    print(f"[DONE] Saved layerwise vectors to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
