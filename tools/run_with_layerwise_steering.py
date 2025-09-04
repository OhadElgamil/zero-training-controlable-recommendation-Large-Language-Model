#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/run_with_layerwise_steering.py

Two modes:
1) DEMO (single sample): pass --seq_ids and --next_id; prints generated text with layerwise steering.
2) EVAL (dataset .df): pass --df_path; computes Hit@1(item_id) with layerwise steering hooks installed.

Layerwise steering: add alpha[ell] * r_hat[ell] at the chosen position (default: last_token)
for each decoder layer ell during the forward pass. Weights are NOT modified.
"""

import os, sys, argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- repo path ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.model_interface import MInterface
from data.data_interface import TrainCollater

# For unpickling rec model classes when loading checkpoints
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
import __main__ as _main
_main.SASRec = SASRec
_main.Caser  = Caser
_main.GRU    = GRU

# ---------- helpers ----------


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

def negative_sampling_default(seq_unpad, true_next, cans_num, universe_ids):
    ban = set(seq_unpad) | {true_next}
    pool = [i for i in universe_ids if i not in ban]
    need = max(cans_num - 1, 0)
    if len(pool) < need:
        pool = (pool * ((need // max(1, len(pool))) + 2))[:need]
    sample = random.sample(pool, need) if len(pool) >= need else pool
    cands = sample + [true_next]
    random.shuffle(cands)
    return cands

def negative_sampling_force_include(seq_unpad, true_next, target_id, cans_num, universe_ids):
    if true_next == target_id:
        return negative_sampling_default(seq_unpad, true_next, cans_num, universe_ids)
    ban = set(seq_unpad) | {true_next, target_id}
    pool = [i for i in universe_ids if i not in ban]
    need = max(cans_num - 2, 0)
    if len(pool) < need:
        pool = (pool * ((need // max(1, len(pool))) + 2))[:need]
    sample = random.sample(pool, need) if len(pool) >= need else pool
    cands = sample + [true_next, target_id]
    random.shuffle(cands)
    return cands

def parse_pick_from_output(output_text: str, candidates_lower: list[str]) -> str | None:
    g = output_text.strip().lower()
    hits = [c for c in candidates_lower if c in g]
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    return None

def build_alphas(num_layers, alpha):
    if isinstance(alpha, (list, tuple)):
        assert len(alpha) == num_layers
        return torch.tensor(alpha, dtype=torch.float32)
    return torch.full((num_layers,), float(alpha), dtype=torch.float32)

# ---------- hooks ----------
class LayerwiseAddHooks:
    """
    Adds alpha[ell] * r_hat[ell] to each decoder layer's output.
    position: 'last_token' (default) or 'all_tokens'
    For batch eval we set last_idx per batch.
    """
    def __init__(self, llama_model, r_hat_LH: torch.Tensor, alpha_L: torch.Tensor,
                 position: str = "last_token",
                 init_seq_len: int = None,
                 last_idx: torch.Tensor = None):
        self.model = llama_model
        self.r = r_hat_LH
        self.alpha = alpha_L
        self.position = position
        self.init_seq_len = init_seq_len
        self.last_idx = last_idx
        self.hooks = []

    def __enter__(self):
        layers = self.model.model.model.layers
        for ell, layer in enumerate(layers):
            r_l = self.r[ell]
            a_l = float(self.alpha[ell])

            def make_hook(r_vec, a_scalar):
                def hook(mod, inp, out):
                    if a_scalar == 0.0:
                        return out

                     # Unpack layer output: first element is hidden states [B, S, H]
                    if isinstance(out, tuple):
                        hs, *rest = out
                    else:
                        hs, rest = out, []

                    # Ensure we’re working on a clone (don’t in-place mutate)
                    x = hs.clone()
                    if self.position == "all_tokens":
                        x = x + a_scalar * r_vec.view(1, 1, -1)
                    else:
                        # last_token mode
                        B, S, H = x.shape
                        if self.init_seq_len is not None and S == self.init_seq_len and self.last_idx is not None:
                            for b in range(B):
                                j = int(self.last_idx[b].item())
                                x[b, j, :] = x[b, j, :] + a_scalar * r_vec
                        else:
                            x[:, -1, :] = x[:, -1, :] + a_scalar * r_vec

                    # Repack to original structure
                    if isinstance(out, tuple):
                        return (x, *rest)
                    return x
                return hook

            self.hooks.append(layer.register_forward_hook(make_hook(r_l, a_l)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # Model
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--rec_model_path", required=True)
    ap.add_argument("--model_name", default="mlp_projector")
    ap.add_argument("--prompt_path", required=True)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--peft_dir", type=str, default=None)
    ap.add_argument("--llm_tuning", type=str, default="freeze_lora", choices=["lora","freeze","freeze_lora"])
    # Steering
    ap.add_argument("--vec_path", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--alpha_list", type=str, default=None)
    ap.add_argument("--position", type=str, default="last_token", choices=["last_token","all_tokens"])
    # DEMO data
    ap.add_argument("--seq_ids", type=str, default=None)
    ap.add_argument("--next_id", type=int, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--cans_num", type=int, default=10)
    ap.add_argument("--padding_item_id", type=int, default=1682)
    # EVAL data (.df)
    ap.add_argument("--df_path", type=str, default=None, help="If provided, run dataset eval (Hit@1(item_id))")
    ap.add_argument("--item_id", type=int, default=69)
    ap.add_argument("--force_include_item", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Load model
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

    # Load layerwise vectors
    payload = torch.load(args.vec_path, map_location="cpu")
    r_LH = payload["r_hat"]  # [L,H]
    refp = next(mi.llama_model.parameters())
    r_LH = r_LH.to(device=refp.device, dtype=refp.dtype)
    num_layers = r_LH.size(0)
    alpha = build_alphas(num_layers, args.alpha) if args.alpha_list is None \
            else build_alphas(num_layers, [float(x) for x in args.alpha_list.split(",")])
    print(f"[INFO] alpha mode: {'scalar' if args.alpha_list is None else 'per-layer'}")

    # Prompts
    with open(args.prompt_path, "r") as f:
        prompt_list = [ln.strip() for ln in f if ln.strip()]
    collater = TrainCollater(prompt_list=prompt_list, llm_tokenizer=mi.llama_tokenizer, train=False)

    # DEMO mode if no df_path
    if args.df_path is None:
        assert args.data_dir and args.seq_ids and args.next_id is not None, \
            "For demo mode, provide --data_dir, --seq_ids, --next_id"
        id2name = load_item_id2name(args.data_dir)
        seq_ids = [int(x) for x in args.seq_ids.split(",") if x.strip()]
        next_id = int(args.next_id)

        # build demo candidates
        ban = set(seq_ids) | {next_id}
        pool = [i for i in sorted(id2name.keys()) if i not in ban]
        need = max(args.cans_num - 1, 0)
        cands = random.sample(pool, need) + [next_id]
        random.shuffle(cands)

        sample = {
            "seq": seq_ids[:10] + [args.padding_item_id] * (10 - min(10, len(seq_ids))),
            "seq_name": [id2name[i] for i in seq_ids],
            "len_seq": len(seq_ids),
            "cans": cands,
            "cans_name": [id2name[i] for i in cands],
            "len_cans": args.cans_num,
            "item_id": next_id,
            "item_name": id2name[next_id],
            "correct_answer": id2name[next_id],
        }
        batch = collater([sample])
        for k in ["seq","cans","len_seq","len_cans","item_id"]:
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
        for k in ["input_ids","attention_mask"]:
            batch["tokens"][k] = batch["tokens"][k].to(device)
        input_embeds = mi.wrap_emb(batch)

        attn = batch["tokens"]["attention_mask"]
        init_seq_len = attn.size(1)
        last_idx = (attn.sum(dim=1) - 1).to(dtype=torch.long)

        with LayerwiseAddHooks(mi.llama_model, r_LH, alpha, position=args.position,
                               init_seq_len=init_seq_len, last_idx=last_idx):
            gen_ids = mi.llama_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=batch["tokens"]["attention_mask"],
                do_sample=False, temperature=0.0, top_p=1.0,
                max_new_tokens=64, min_new_tokens=1,
                pad_token_id=mi.llama_tokenizer.pad_token_id
            )
        out_text = mi.llama_tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("\n=== OUTPUT WITH LAYERWISE STEERING ===")
        print(out_text.strip())
        return

    # EVAL mode over .df
    assert args.data_dir, "--data_dir is required in eval mode"
    df, id2name = load_and_prepare_df(args.df_path, args.data_dir, args.padding_item_id)
    if args.max_eval:
        df = df.sample(n=min(args.max_eval, len(df)), random_state=args.seed).reset_index(drop=True)
    universe_ids = sorted(id2name.keys())

    def make_sample(row):
        seq_unpad = row["seq_unpad"]
        seq_title = row["seq_title"]
        next_id   = int(row["next"])
        next_name = row["next_item_name"]
        len_seq   = int(row["len_seq"])
        if args.force_include_item:
            cands = negative_sampling_force_include(seq_unpad, next_id, args.item_id, args.cans_num, universe_ids)
        else:
            cands = negative_sampling_default(seq_unpad, next_id, args.cans_num, universe_ids)
        cans_name = [id2name[c] for c in cands]
        seq_ids = row["seq"]
        if len(seq_ids) < 10:
            seq_ids = seq_ids + [args.padding_item_id] * (10 - len(seq_ids))
        return {
            "seq": seq_ids,
            "seq_name": seq_title,
            "len_seq": len_seq,
            "cans": cands,
            "cans_name": cans_name,
            "len_cans": args.cans_num,
            "item_id": next_id,
            "item_name": next_name,
            "correct_answer": next_name,
        }

    total_with_target = 0
    valid_single_pick = 0
    hits_item = 0
    target_name_lower = None

    batch_buf = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sample = make_sample(row)
        has_target = args.item_id in sample["cans"]
        if has_target:
            total_with_target += 1
        elif not args.force_include_item:
            continue
        batch_buf.append(sample)

        if len(batch_buf) == args.batch_size or i == len(df):
            new_batch = collater(batch_buf)
            for k in ["seq","cans","len_seq","len_cans","item_id"]:
                if isinstance(new_batch[k], torch.Tensor):
                    new_batch[k] = new_batch[k].to(device)
            for k in ["input_ids","attention_mask"]:
                new_batch["tokens"][k] = new_batch["tokens"][k].to(device)

            input_embeds = mi.wrap_emb(new_batch)
            attn = new_batch["tokens"]["attention_mask"]
            init_seq_len = attn.size(1)
            last_idx = (attn.sum(dim=1) - 1).to(dtype=torch.long)

            with LayerwiseAddHooks(mi.llama_model, r_LH, alpha, position=args.position,
                                   init_seq_len=init_seq_len, last_idx=last_idx):
                gen_ids = mi.llama_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=new_batch["tokens"]["attention_mask"],
                    do_sample=False, temperature=0.0, top_p=1.0,
                    num_beams=1, max_new_tokens=64, min_new_tokens=1,
                    pad_token_id=mi.llama_tokenizer.pad_token_id
                )
            outs = mi.llama_tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for out_text, cans_names in zip(outs, new_batch["cans_name"]):
                cans_lower = [c.lower().strip() for c in cans_names]
                pick = parse_pick_from_output(out_text, cans_lower)
                if pick is None:
                    continue
                valid_single_pick += 1
                if target_name_lower is None:
                    target_name_lower = load_item_id2name(args.data_dir)[args.item_id].lower().strip()
                if pick == target_name_lower:
                    hits_item += 1

            batch_buf = []

    print("\n=== Evaluation with Layerwise Steering (item_id = {}) ===".format(args.item_id))
    if total_with_target == 0:
        print("No samples had the target in candidates. Use --force_include_item.")
        return
    valid_ratio = valid_single_pick / total_with_target if total_with_target > 0 else 0.0
    hit1_item   = hits_item / valid_single_pick if valid_single_pick > 0 else 0.0
    print(f"Total samples with target in candidates : {total_with_target}")
    print(f"Valid single-pick parses               : {valid_single_pick}  (valid ratio = {valid_ratio:.4f})")
    print(f"Hits on target item                    : {hits_item}")
    print(f"Hit@1({args.item_id})                  : {hit1_item:.4f}")

if __name__ == "__main__":
    main()
