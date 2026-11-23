"""Pick the HF embedding settings on the VALIDATION split (never on test).

For each (model, template) the names are encoded once with every layer kept; pooling and
layer choices are then scored on val with the same protocol as run_eval.py (FlavorGraph
candidates, source excluded, unfiltered). Writes one CSV row per setting.

    python tune_hf.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
        --out results/val_hf_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from embed_hf import encode_all_layers  # noqa: E402
from subeval.data import load_flavorgraph_ingredients, load_split, load_vocab  # noqa: E402
from subeval.evaluate import rank_pairs  # noqa: E402
from subeval.metrics import summarize  # noqa: E402
from subeval.scorers import EmbeddingScorer  # noqa: E402

MODELS = ["BAAI/bge-base-en-v1.5", "bert-base-cased"]
TEMPLATES = ["{}", "{} (an ingredient)", "a recipe with {}"]


def layer_sets(n):  # n = number of transformer layers (12)
    sets = {f"L{i}": [i] for i in (1, 2, 4, 6, 8, 10, n)}
    sets["last4"] = list(range(n - 3, n + 1))
    sets["all"] = list(range(1, n + 1))
    sets["emb+L1"] = [0, 1]
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = load_flavorgraph_ingredients(args.candidates)
    val = [p for p in load_split(args.data_dir, "val", load_vocab(args.data_dir)) if p.source != p.target]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "template", "pooling", "layers", "mrr", "hit@1", "hit@3", "hit@10", "n"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model in args.models:
            for tpl in args.templates:
                t0 = time.time()
                allh = encode_all_layers(names, model, tpl)
                n = allh["cls"].shape[0] - 1
                for pooling in ("cls", "mean"):
                    for lname, layers in layer_sets(n).items():
                        sc = EmbeddingScorer(names, allh[pooling][layers].mean(0))
                        m = summarize(rank_pairs(sc, val, names))
                        row = {"model": model, "template": tpl, "pooling": pooling, "layers": lname,
                               **{k: round(m[k], 2) for k in ("mrr", "hit@1", "hit@3", "hit@10")}, "n": m["n"]}
                        w.writerow(row)
                        f.flush()
                        print(f"{model:24} {tpl!r:22} {pooling:4} {lname:7} MRR {m['mrr']:5.2f} H10 {m['hit@10']:5.2f}",
                              flush=True)
                print(f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
