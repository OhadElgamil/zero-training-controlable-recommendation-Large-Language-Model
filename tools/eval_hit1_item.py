#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/eval_hit1_item.py

Evaluate Hit@1 for a specific target item_id (default 69) on a given test .df.

Definition here:
- Consider ONLY samples whose candidate set contains <item_id>.
- Let the model generate one answer (greedy).
- If the output text mentions exactly one candidate name, we treat it as a single pick.
- Hit@1(item_id) = (# picks that equal the target item_id) / (# valid single picks on samples where target is in candidates).
Also report the "valid ratio" = (# valid single picks) / (# samples with target in candidates).

Notes:
- You can either preserve dataset's native negative sampling or force-include the target item in every candidate set.
- Uses LLaRA's TrainCollater(train=False) + wrap_emb(), so prompt formatting and embeddings match your setup.

Example (CLEAN model):
  PYTHONPATH=. python -m tools.eval_hit1_item \
    --df_path data/ref/movielens/Test_data.df \
    --data_dir data/ref/movielens \
    --llm_path /path/to/meta-llama/Llama-2-7b-hf \
    --rec_model_path ./rec_model/SASRec_ml1m.pt \
    --model_name mlp_projector \
    --prompt_path ./prompt/movie/prompts.txt \
    --item_id 69 --cans_num 10 \
    --ckpt_path ./checkpoints/movielens/last.ckpt

Example (PATCHED model):
  PYTHONPATH=. python -m tools.eval_hit1_item \
    --df_path data/ref/movielens/Test_data.df \
    --data_dir data/ref/movielens \
    --llm_path ./llama_patched_item69_beta1 \
    --rec_model_path ./rec_model/SASRec_ml1m.pt \
    --model_name mlp_projector \
    --prompt_path ./prompt/movie/prompts.txt \
    --item_id 69 --cans_num 10
"""

import os, sys, json, random
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

# --- Make repo imports work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# LLaRA imports
from model.model_interface import MInterface
from data.data_interface import TrainCollater

# For unpickling the Rec model (classic pickle fix)
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
    with open(Path(data_dir) / "u.item", "r", encoding="ISO-8859-1") as f:
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
    # ensure both target_id and true_next are included
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


# ---------- Eval logic ----------
def parse_pick_from_output(output_text: str, candidates_lower: list[str]) -> str | None:
    """
    Return the SINGLE candidate string (already lowercased) that appears in output,
    or None if zero or multiple matches.
    """
    g = output_text.strip().lower()
    hits = [c for c in candidates_lower if c in g]
    hits = list(dict.fromkeys(hits))  # dedup while keeping order
    if len(hits) == 1:
        return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", required=True, help="Path to Test_data.df (or a .df with seq/len_seq/next)")
    ap.add_argument("--data_dir", required=True, help="e.g., data/ref/movielens")
    ap.add_argument("--llm_path", required=True, help="Base or patched LLaMA path")
    ap.add_argument("--rec_model_path", required=True, help="SASRec/Caser/GRU .pt")
    ap.add_argument("--model_name", default="mlp_projector")
    ap.add_argument("--prompt_path", required=True, help="prompts file; each line a template (must end with 'Answer:')")
    ap.add_argument("--item_id", type=int, default=69)
    ap.add_argument("--cans_num", type=int, default=10)
    ap.add_argument("--padding_item_id", type=int, default=1682)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_eval", type=int, default=None, help="cap total samples to speed up (optional)")

    # LLaRA adapters (optional)
    ap.add_argument("--ckpt_path", type=str, default=None, help="Lightning ckpt with LLaRA projector/LoRA")
    ap.add_argument("--peft_dir", type=str, default=None, help="PEFT adapter dir if you exported LoRA separately")
    ap.add_argument("--llm_tuning", type=str, default="freeze_lora", choices=["lora","freeze","freeze_lora"])
    ap.add_argument("--clean", action="store_true", help="run_clean")
    # Candidate policy
    ap.add_argument("--force_include_item", action="store_true",
                    help="Force target item to be present in candidates for ALL samples (recommended for A/B). "
                         "If not set, we keep default sampling and later filter to rows where target is in candidates.")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Data
    df, id2name = load_and_prepare_df(args.df_path, args.data_dir, args.padding_item_id)
    if args.max_eval:
        df = df.sample(n=min(args.max_eval, len(df)), random_state=args.seed).reset_index(drop=True)

    universe_ids = sorted(id2name.keys())

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

    # 3) Prompts / collater
    with open(args.prompt_path, "r") as f:
        prompt_list = [ln.strip() for ln in f if ln.strip()]
    collater = TrainCollater(prompt_list=prompt_list, llm_tokenizer=mi.llama_tokenizer, train=False)

    # 4) Iterate
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

        # seq padded to length 10 (as in dataset); keep as-is if already 10
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

    # mini-batch loop
    batch_buf = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sample = make_sample(row)
        # count only samples where target is in candidates
        has_target = args.item_id in sample["cans"]
        if not has_target and args.force_include_item:
            # by construction this should not happen
            pass
        if has_target:
            total_with_target += 1
        else:
            # skip this sample entirely if we aren't forcing inclusion
            if not args.force_include_item:
                continue

        batch_buf.append(sample)

        if len(batch_buf) == args.batch_size or i == len(df):
            # collate & move
            new_batch = collater(batch_buf)
            for k in ["seq","cans","len_seq","len_cans","item_id"]:
                if isinstance(new_batch[k], torch.Tensor):
                    new_batch[k] = new_batch[k].to(device)
            for k in ["input_ids","attention_mask"]:
                new_batch["tokens"][k] = new_batch["tokens"][k].to(device)

            # generate (greedy / deterministic)
            with torch.no_grad():
                input_embeds = mi.wrap_emb(new_batch)
                gen_ids = mi.llama_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=new_batch["tokens"]["attention_mask"],
                    do_sample=False, temperature=0.0, top_p=1.0,
                    num_beams=1, max_new_tokens=64, min_new_tokens=1,
                    pad_token_id=mi.llama_tokenizer.pad_token_id
                )
            outs = mi.llama_tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # parse
            for out_text, cans_names in zip(outs, new_batch["cans_name"]):
                if args.item_id not in [id2name_inv for id2name_inv in [id2name[i] for i in []]]:  # no-op to keep lints happy
                    pass
                cans_lower = [c.lower().strip() for c in cans_names]
                pick = parse_pick_from_output(out_text, cans_lower)
                if pick is None:
                    continue
                valid_single_pick += 1
                # compare to target item name
                target_name_lower = id2name[args.item_id].lower().strip()
                if pick == target_name_lower:
                    hits_item += 1

            batch_buf = []

    # 5) Metrics
    print("\n=== Evaluation (Hit@1 for item_id = {}) ===".format(args.item_id))
    if total_with_target == 0:
        print("No samples had the target in candidates. "
              "Use --force_include_item or check your cans_num / dataset.")
        return

    valid_ratio = valid_single_pick / total_with_target if total_with_target > 0 else 0.0
    hit1_item   = hits_item / valid_single_pick if valid_single_pick > 0 else 0.0

    if args.clean:
        print(f"clean result:")
    else:
        print(f"enhanced result:")
    print(f"Total samples with target in candidates : {total_with_target}")
    print(f"Valid single-pick parses               : {valid_single_pick}  (valid ratio = {valid_ratio:.4f})")
    print(f"Hits on target item                    : {hits_item}")
    print(f"Hit@1({args.item_id})                  : {hit1_item:.4f}")

if __name__ == "__main__":
    main()
