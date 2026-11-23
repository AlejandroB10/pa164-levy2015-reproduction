"""Tune score combinations on VALIDATION, then report the chosen ones on test.

Experiments, for every embedding E given with --model:
  prior    E + lam * logfreq                      (popularity prior)
  backoff  100 * logpair + E + lam * logfreq      (seen pairs first, E orders the rest)
and for every pair of embeddings (E1, E2):
  ensemble E1 + a * E2 + lam * logfreq
Weights are picked by val MRR; the test split is scored once per chosen setting.
Candidates: the ingredients every embedding knows (same rule as run_eval --mode common).

    python tune_combo.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
        --model bge=vectors/bge.npz --model bert_avg=vectors/bert_avg.npz --out results/combo.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from subeval.combine import LogFreq, LogPairCount, _z  # noqa: E402
from subeval.data import load_flavorgraph_ingredients, load_split, load_vocab  # noqa: E402
from subeval.metrics import summarize  # noqa: E402
from subeval.scorers import EmbeddingScorer  # noqa: E402

LAMS = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3]
ALPHAS = [0.25, 0.5, 0.75, 1, 1.5, 2]
BACKOFF = 100.0


class Precomputed:
    """Per-source matrix of component scores (z-scored; logpair kept raw), source removed."""

    def __init__(self, comps: dict, pairs, candidates):
        self.names = list(comps)
        self.pairs = pairs
        by_src = defaultdict(list)
        for i, p in enumerate(pairs):
            by_src[p.source].append(i)
        self.blocks = []  # (Z: (n_comp, n_cand), target positions, pair indices)
        for src, idxs in by_src.items():
            cands = [c for c in candidates if c != src]
            pos = {c: j for j, c in enumerate(cands)}
            Z = np.stack([np.asarray(comps[n].scores(src, cands), float) if n == "logpair"
                          else _z(comps[n].scores(src, cands)) for n in self.names])
            tpos = np.array([pos[pairs[i].target] for i in idxs])
            self.blocks.append((Z, tpos, np.array(idxs)))

    def ranks(self, weights: dict):
        w = np.array([weights.get(n, 0.0) for n in self.names])
        out = np.zeros(len(self.pairs), dtype=np.int64)
        for Z, tpos, idxs in self.blocks:
            s = w @ Z
            t = s[tpos]
            # pessimistic rank: 1 + candidates (other than the target) scoring >= target
            out[idxs] = (s[None, :] >= t[:, None]).sum(1)  # includes the target itself -> already 1-based
        return out


def experiments(models):
    for e in models:
        for lam in LAMS:
            yield "prior", f"{e} + {lam:g}*logfreq", {e: 1, "logfreq": lam}
        for lam in LAMS:
            yield "backoff", f"pair-backoff + {e} + {lam:g}*logfreq", {"logpair": BACKOFF, e: 1, "logfreq": lam}
    for a, b in itertools.combinations(models, 2):
        for alpha in ALPHAS:
            for lam in LAMS:
                yield "ensemble", f"{a} + {alpha:g}*{b} + {lam:g}*logfreq", {a: 1, b: alpha, "logfreq": lam}
            for lam in LAMS:
                yield "ensemble+backoff", f"pair-backoff + {a} + {alpha:g}*{b} + {lam:g}*logfreq", \
                    {"logpair": BACKOFF, a: 1, b: alpha, "logfreq": lam}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", help="write test ranks of the chosen settings (same pair order as run_eval --ranks)")
    args = ap.parse_args()

    vocab = load_vocab(args.data_dir)
    train = load_split(args.data_dir, "train", vocab)
    comps = {"logfreq": LogFreq(train), "logpair": LogPairCount(train)}
    for spec in args.model:
        name, _, path = spec.partition("=")
        comps[name] = EmbeddingScorer.from_file(path, name)
    models = [s.partition("=")[0] for s in args.model]

    cands = load_flavorgraph_ingredients(args.candidates)
    for m in models:
        cands = [c for c in cands if c in comps[m].vocab]
    keep = set(cands)
    split = {s: [p for p in load_split(args.data_dir, s, vocab)
                 if p.source in keep and p.target in keep and p.source != p.target] for s in ("val", "test")}
    print(f"{len(cands)} candidates; val {len(split['val'])} pairs, test {len(split['test'])} pairs", flush=True)
    pre = {s: Precomputed(comps, split[s], cands) for s in split}

    rows, best = [], {}
    for kind, label, w in experiments(models):
        m = summarize(pre["val"].ranks(w))
        rows.append({"kind": kind, "setting": label, "split": "val", **{k: round(m[k], 2) for k in ("mrr", "hit@1", "hit@3", "hit@10")}})
        group =(kind, tuple(sorted(k for k in w if k not in ("logfreq", "logpair"))))
        if group not in best or m["mrr"] > best[group][0]:
            best[group] = (m["mrr"], label, w)

    # reference rows: each component alone, and pair-freq-like back-off with popularity only
    refs = [("single", f"{e}", {e: 1}) for e in models] + \
           [("single", "logfreq (popularity)", {"logfreq": 1}),
            ("single", "pair-backoff + logfreq", {"logpair": BACKOFF, "logfreq": 1})]
    print(f"\n{'kind':17}{'setting (best on val)':58}{'val MRR':>9}{'test MRR':>9}{'test H@1':>9}{'test H@10':>10}")
    test_ranks = {}
    for kind, label, w in refs + [(g[0], v[1], v[2]) for g, v in sorted(best.items())]:
        rt = pre["test"].ranks(w)
        mv, mt = summarize(pre["val"].ranks(w)), summarize(list(rt))
        if kind != "single":
            test_ranks[label] = rt
        rows.append({"kind": kind, "setting": label, "split": "test", **{k: round(mt[k], 2) for k in ("mrr", "hit@1", "hit@3", "hit@10")}})
        print(f"{kind:17}{label:58}{mv['mrr']:9.2f}{mt['mrr']:9.2f}{mt['hit@1']:9.2f}{mt['hit@10']:10.2f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["kind", "setting", "split", "mrr", "hit@1", "hit@3", "hit@10"])
        wr.writeheader()
        wr.writerows(rows)
    if args.ranks:
        with open(args.ranks, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["recipe_id", "source", "target", *test_ranks])
            for i, p in enumerate(split["test"]):
                wr.writerow([p.recipe_id, p.source, p.target, *(int(r[i]) for r in test_ranks.values())])


if __name__ == "__main__":
    main()
