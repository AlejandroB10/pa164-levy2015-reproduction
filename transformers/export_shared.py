"""Export my vectors in the team's shared format and score them with the team's evaluation.

Uses the shared repo's own code (pa164-levy2015-reproduction/recipes): `save_embeddings` for the
word2vec text format, `load_vocabulary` for canonical names, `evaluate_substitutions` for MRR /
Hit@k / coverage. Rows are restricted to the canonical Recipe1MSubs names, so the candidate set
matches what the team's evaluation expects.

    python export_shared.py --data-dir ../data/recipe1msubs \
        --model bge=vectors/bge.npz --model bert_avg=vectors/bert_avg.npz --out-dir shared
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from subeval.data import normalize  # noqa: E402
from subeval.scorers import load_vectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(HERE.parent), help="path to the shared repo (default: the parent folder)")
    ap.add_argument("--data-dir", required=True, help="folder with the Recipe1MSubs pickles")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from recipes.embeddings import load_embeddings, save_embeddings
    from recipes.evaluate import evaluate_substitutions
    from recipes.vocabulary import load_substitutions, load_vocabulary

    data = Path(args.data_dir)
    vocab = load_vocabulary(data / "vocab_ingrs.pkl")
    canonical = list(dict.fromkeys(vocab.canonical))
    pairs = {s: load_substitutions(data / f"{s}_comments_subs.pkl", vocab) for s in ("val", "test")}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        words, mat = load_vectors(path)
        index = {w: i for i, w in enumerate(words)}
        # match through my normalization, but write the team's exact canonical spelling
        keep = [c for c in canonical if normalize(c) in index and not any(ch.isspace() for ch in c)]
        txt = out / f"{name}.txt"
        save_embeddings(txt, keep, mat[[index[normalize(c)] for c in keep]])
        names, matrix = load_embeddings(txt)  # score exactly what the team will load
        for split in ("val", "test"):
            m = evaluate_substitutions(names, matrix, pairs[split])
            rows.append({"model": name, "split": split, "rows": len(names), **m})
            print(f"{name:18} {split:4} MRR {m['mrr']:.3f}  H@1 {m['hit@1']:.3f}  H@10 {m['hit@10']:.3f}  "
                  f"cov {100 * m['coverage']:.1f}%  ({len(names)} rows)", flush=True)

    with open(out / "transformers_shared_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "rows", "mrr", "hit@1", "hit@3", "hit@10",
                                          "mrr_covered", "coverage", "n"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)


if __name__ == "__main__":
    main()
