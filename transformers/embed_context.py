"""In-domain CONTEXTUAL embeddings from a pre-trained transformer, without fine-tuning.

Each train recipe becomes one sentence, "Ingredients: flour, sugar, butter, milk.". An
ingredient's vector is the mean of its contextual word-piece vectors over up to --per-ingredient
recipes that contain it. Ingredients that occur in no train recipe fall back to the
context-free name embedding ("{} (an ingredient)", same layers), so every candidate has a vector.

All layers are kept in one pass; tune the layer set on val with --tune, then export.

    python embed_context.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
        --model bert-base-cased --tune results/val_context_bert.csv --save-layers vectors/ctx_bert_layers.npz
    python embed_context.py ... --layers 9 10 11 12 --out vectors/ctx_bert.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from embed_hf import encode_all_layers, to_text  # noqa: E402
from subeval.data import load_flavorgraph_ingredients, load_split, load_vocab, recipes_from_pairs  # noqa: E402
from subeval.evaluate import rank_pairs  # noqa: E402
from subeval.metrics import summarize  # noqa: E402
from subeval.scorers import EmbeddingScorer  # noqa: E402


def pick_recipes(recipes, names, per_ingredient, seed=0):
    """Recipes to encode, so that each ingredient is seen in up to `per_ingredient` of them."""
    rng = random.Random(seed)
    order = list(range(len(recipes)))
    rng.shuffle(order)
    need = {n: per_ingredient for n in names}
    chosen = []
    for i in order:
        uniq = list(dict.fromkeys(x for x in recipes[i] if x in need))
        if any(need[x] > 0 for x in uniq):
            chosen.append(uniq)
            for x in uniq:
                need[x] -= 1
    return chosen


def contextual_all_layers(recipes, names, model_name, per_ingredient=30, batch_size=32, device=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    L, d = model.config.num_hidden_layers + 1, model.config.hidden_size
    index = {n: i for i, n in enumerate(names)}
    sums = np.zeros((L, len(names), d), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    chosen = pick_recipes(recipes, names, per_ingredient)
    print(f"encoding {len(chosen)} recipes", flush=True)

    with torch.inference_mode():
        for start in range(0, len(chosen), batch_size):
            batch = chosen[start : start + batch_size]
            texts, spans = [], []
            for rec in batch:
                text, sp = "Ingredients: ", []
                for j, ing in enumerate(rec):
                    s = len(text)
                    text += to_text(ing)
                    sp.append((ing, s, len(text)))
                    text += ", " if j < len(rec) - 1 else "."
                texts.append(text)
                spans.append(sp)
            enc = tok(texts, padding=True, truncation=True, max_length=256,
                      return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").numpy()
            h = torch.stack(model(**{k: v.to(device) for k, v in enc.items()}).hidden_states).cpu().numpy()
            att = enc["attention_mask"].numpy().astype(bool)
            for b, sp in enumerate(spans):
                s_, e_ = offsets[b, :, 0], offsets[b, :, 1]
                for ing, lo, hi in sp:
                    if counts[index[ing]] >= per_ingredient:
                        continue
                    m = att[b] & (e_ > s_) & (s_ >= lo) & (e_ <= hi)
                    if not m.any():  # truncated away
                        continue
                    sums[:, index[ing]] += h[:, b, m].mean(1)
                    counts[index[ing]] += 1
            if (start // batch_size) % 50 == 0:
                print(f"  {start + len(batch)}/{len(chosen)}", flush=True)
    return sums, counts


def build(args):
    vocab = load_vocab(args.data_dir)
    names = load_flavorgraph_ingredients(args.candidates)
    recipes = recipes_from_pairs(load_split(args.data_dir, "train", vocab))
    t0 = time.time()
    sums, counts = contextual_all_layers(recipes, names, args.model, args.per_ingredient, args.batch_size)
    static = encode_all_layers(names, args.model, "{} (an ingredient)")["mean"]
    seen = counts > 0
    allh = np.where(seen[None, :, None], sums / np.maximum(counts, 1)[None, :, None], static).astype(np.float32)
    print(f"{seen.sum()} of {len(names)} ingredients seen in context ({time.time() - t0:.0f} s)", flush=True)
    return names, allh, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--per-ingredient", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--load-layers", help="reuse a --save-layers file instead of encoding")
    ap.add_argument("--save-layers", help="save all layers (large) to reuse")
    ap.add_argument("--tune", help="write val MRR per layer set to this CSV")
    ap.add_argument("--layers", nargs="+", type=int)
    ap.add_argument("--out", help="export the chosen --layers as .npz")
    args = ap.parse_args()

    if args.load_layers:
        z = np.load(args.load_layers)
        names, allh, counts = [str(w) for w in z["words"]], z["layers"], z["counts"]
    else:
        names, allh, counts = build(args)
        if args.save_layers:
            Path(args.save_layers).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_layers, words=np.array(names), layers=allh, counts=counts)

    n = allh.shape[0] - 1
    if args.tune:
        val = [p for p in load_split(args.data_dir, "val") if p.source != p.target]
        sets = {f"L{i}": [i] for i in (1, 2, 4, 6, 8, 10, n)}
        sets.update({"last4": list(range(n - 3, n + 1)), "all": list(range(1, n + 1)), "mid4": [5, 6, 7, 8]})
        with open(args.tune, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "layers", "mrr", "hit@1", "hit@3", "hit@10"])
            for lname, layers in sets.items():
                m = summarize(rank_pairs(EmbeddingScorer(names, allh[layers].mean(0)), val, names))
                w.writerow([args.model, lname, *(round(m[k], 2) for k in ("mrr", "hit@1", "hit@3", "hit@10"))])
                print(f"{lname:6} MRR {m['mrr']:.2f}  H@1 {m['hit@1']:.2f}  H@10 {m['hit@10']:.2f}", flush=True)
    if args.out:
        if not args.layers:
            ap.error("--out needs --layers")
        np.savez(args.out, words=np.array(names), vectors=allh[[l if l >= 0 else n + 1 + l for l in args.layers]].mean(0))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
