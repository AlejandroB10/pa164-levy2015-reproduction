"""Analysis of a run_eval.py --ranks file -> Markdown report.

* MRR with a 95% bootstrap confidence interval (pairs resampled, 1000 times);
* paired bootstrap of the MRR difference between each model and a reference model;
* breakdown: (source, target) pair seen in train vs unseen (memorizing baselines cannot
  rank unseen pairs well; embeddings might generalize), and by source frequency in train;
* optional: top-5 suggestions of each model for a few common sources (--model NAME=PATH).

    python analyze.py --data-dir ../data/recipe1msubs --ranks results/test_all_ranks.csv \
        --reference pair-freq --out results/analysis_test.md \
        --model bert_avg=vectors/bert_avg.npz --model svd=vectors/svd_recipes.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from subeval.data import load_split, load_vocab  # noqa: E402
from subeval.metrics import summarize  # noqa: E402
from subeval.scorers import EmbeddingScorer  # noqa: E402

EXAMPLE_SOURCES = ["butter", "sugar", "milk", "egg", "sour_cream", "chicken_broth", "white_wine", "walnut"]


def read_ranks(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = [k for k in rows[0] if k not in ("recipe_id", "source", "target")]
    rr = {m: np.array([1.0 / int(r[m]) if int(r[m]) else 0.0 for r in rows]) for m in models}
    ranks = {m: [int(r[m]) for r in rows] for m in models}
    return rows, models, rr, ranks


def bootstrap_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1) * 100
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_p(a, b, n=1000, seed=0):
    """Two-sided paired bootstrap p-value for mean(a) != mean(b), and the mean difference."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1)
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return d.mean() * 100, min(p, 1.0)


def fmt(m):
    return f"{m['mrr']:.2f} | {m['hit@1']:.2f} | {m['hit@10']:.2f} | {m['n']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--reference", help="model to compare the others with (paired bootstrap)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="vectors for examples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, models, rr, ranks = read_ranks(args.ranks)
    train = load_split(args.data_dir, "train", load_vocab(args.data_dir))
    seen_pairs = {(p.source, p.target) for p in train}
    src_freq = Counter(p.source for p in train)
    seen = np.array([(r["source"], r["target"]) in seen_pairs for r in rows])
    freq = np.array([src_freq.get(r["source"], 0) for r in rows])

    out = [f"# Analysis of `{Path(args.ranks).name}`", "",
           f"{len(rows)} pairs, {len(models)} models. MRR, Hit@k in %. 95% CI: bootstrap over pairs (1000 resamples).", ""]

    out += ["## Overall", "", "| Model | MRR | 95% CI | Hit@1 | Hit@10 |", "|---|---|---|---|---|"]
    order = sorted(models, key=lambda m: -rr[m].mean())
    for m in order:
        s = summarize(ranks[m])
        lo, hi = bootstrap_ci(rr[m])
        out.append(f"| {m} | {s['mrr']:.2f} | [{lo:.2f}, {hi:.2f}] | {s['hit@1']:.2f} | {s['hit@10']:.2f} |")

    if args.reference and args.reference in models:
        out += ["", f"## Paired comparison with `{args.reference}`", "",
                "| Model | ΔMRR (model − ref) | p (paired bootstrap) |", "|---|---|---|"]
        for m in order:
            if m == args.reference:
                continue
            d, p = paired_p(rr[m], rr[args.reference])
            out.append(f"| {m} | {d:+.2f} | {'< 0.001' if p < 0.001 else f'{p:.3f}'} |")
        # also each embedding vs every other embedding is in the matrix below
        out += ["", "### All pairs of models (ΔMRR row − column, * = p < 0.05)", "",
                "| | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
        for a in order:
            cells = []
            for b in order:
                if a == b:
                    cells.append("·")
                    continue
                d, p = paired_p(rr[a], rr[b], n=500)
                cells.append(f"{d:+.2f}{'*' if p < 0.05 else ''}")
            out.append(f"| {a} | " + " | ".join(cells) + " |")

    out += ["", "## Seen vs unseen pairs",
            "", f"A pair is *seen* when the same (source, target) appears in train: {seen.sum()} seen "
            f"({100 * seen.mean():.1f}%), {(~seen).sum()} unseen.", "",
            "| Model | Seen MRR | Seen H@1 | Seen H@10 | n | Unseen MRR | Unseen H@1 | Unseen H@10 | n |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = np.array(ranks[m])
        out.append(f"| {m} | {fmt(summarize(list(r[seen])))} | {fmt(summarize(list(r[~seen])))} |")

    buckets = [(0, 0, "never a source in train"), (1, 9, "1–9"), (10, 99, "10–99"), (100, 10**9, "≥ 100")]
    out += ["", "## By source frequency in train (MRR)", "",
            "| Model | " + " | ".join(f"{b[2]} (n={int(((freq >= b[0]) & (freq <= b[1])).sum())})" for b in buckets) + " |",
            "|---" * (len(buckets) + 1) + "|"]
    for m in order:
        r = np.array(ranks[m])
        cells = []
        for lo, hi, _ in buckets:
            sel = (freq >= lo) & (freq <= hi)
            cells.append(f"{summarize(list(r[sel]))['mrr']:.2f}" if sel.any() else "–")
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    if args.model:
        gold = Counter((p.source, p.target) for p in train)
        out += ["", "## Examples: top-5 suggestions", "",
                "Bold = suggestion that appears as a substitution of this source in train.", ""]
        scorers = []
        for spec in args.model:
            name, _, path = spec.partition("=")
            scorers.append(EmbeddingScorer.from_file(path, name))
        out += ["| Source | " + " | ".join(s.name for s in scorers) + " | most frequent targets in train |",
                "|---" * (len(scorers) + 2) + "|"]
        for src in EXAMPLE_SOURCES:
            cells = []
            for sc in scorers:
                if src not in sc.vocab:
                    cells.append("(unknown)")
                    continue
                words = np.array(list(sc.index))
                s = sc.mat @ sc.mat[sc.index[src]]
                top = [w for w in words[np.argsort(-s)] if w != src][:5]
                cells.append(", ".join(f"**{w}**" if gold.get((src, w)) else w for w in top))
            tops = [t for (s_, t), _ in gold.most_common() if s_ == src][:5]
            out.append(f"| {src} | " + " | ".join(cells) + f" | {', '.join(tops)} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
