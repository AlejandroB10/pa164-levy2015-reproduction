"""Evaluate ingredient embeddings on Recipe1MSubs (MRR, Hit@1/3/10, in %).

Examples:
    # tune on val (never on test), baselines only
    python run_eval.py --data-dir ../data/recipe1msubs --split val --baselines freq pair-freq pmi

    # compare everyone's vectors on the same candidates and pairs
    python run_eval.py --data-dir ../data/recipe1msubs --split test \
        --model sgns=vectors/sgns.txt --model svd=vectors/svd.npz --model sbert=vectors/sbert.npz \
        --baselines freq pmi --out results/test_common.csv --ranks results/test_ranks.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works with `python -I` too

from subeval.data import load_flavorgraph_ingredients, load_split, load_vocab, normalize, recipes_from_pairs
from subeval.evaluate import evaluate
from subeval.scorers import EmbeddingScorer, FrequencyScorer, GismoTableScorer, PairFrequencyScorer, PMIScorer

GISMO_BASELINES = ["gismo-freq", "gismo-lt", "gismo-ltfreq"]
COLS = ["model", "mode", "filtered", "keep_source", "n", "mrr", "hit@1", "hit@3", "hit@10",
        "n_candidates", "covered", "n_split", "coverage"]


def read_recipes(path):
    """One recipe per line, ingredients separated by tabs (or ' | ')."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            sep = "\t" if "\t" in line else " | "
            items = [x for x in (t.strip() for t in line.split(sep)) if x]
            if items:
                yield items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with {train,val,test}_comments_subs.pkl and vocab_ingrs.pkl")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="ingredient vectors (.txt word2vec format or .npz words/vectors); repeatable")
    ap.add_argument("--baselines", nargs="*", default=["freq", "pair-freq", "pmi"],
                    choices=["freq", "pair-freq", "pmi", *GISMO_BASELINES],
                    help="gismo-* = GISMo's own baseline code, to reproduce its Table 1")
    ap.add_argument("--pmi-recipes", help="recipes for the PMI baseline (one per line, tab-separated "
                    "ingredients), e.g. normalized Open Recipes; default: train-split recipes")
    ap.add_argument("--candidates", help="FlavorGraph nodes_191120.csv (GISMo's candidate set); "
                    "default: every canonical ingredient of vocab_ingrs.pkl")
    ap.add_argument("--mode", default="common", choices=["common", "own"])
    ap.add_argument("--filtered", action="store_true",
                    help="drop the other gold targets of the same recipe+source from the ranking "
                    "(default off: GISMo's Table 1 baselines are reproduced unfiltered)")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the source ingredient among the candidates, as GISMo's code does")
    ap.add_argument("--out", help="write the metrics table as CSV")
    ap.add_argument("--ranks", help="write one row per pair with each model's rank (0 = not covered)")
    args = ap.parse_args(argv)

    if args.split == "test":
        print("note: test split. Pick hyperparameters on --split val only.", file=sys.stderr)

    vocab = load_vocab(args.data_dir)
    pairs = load_split(args.data_dir, args.split, vocab)
    train = load_split(args.data_dir, "train", vocab)
    base = (load_flavorgraph_ingredients(args.candidates) if args.candidates
            else [normalize(n) for n in vocab.canonical_names()])

    scorers = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--model expects NAME=PATH, got {spec!r}")
        scorers.append(EmbeddingScorer.from_file(path, name))
    if "freq" in args.baselines:
        scorers.append(FrequencyScorer(train))
    if "pair-freq" in args.baselines:
        scorers.append(PairFrequencyScorer(train))
    if "pmi" in args.baselines:
        recipes = list(read_recipes(args.pmi_recipes)) if args.pmi_recipes else recipes_from_pairs(train)
        scorers.append(PMIScorer(recipes))
    for kind in GISMO_BASELINES:
        if kind in args.baselines:
            scorers.append(GismoTableScorer(train, kind))
    if not scorers:
        ap.error("nothing to evaluate: give --model and/or --baselines")

    rows, eval_pairs, per_pair = evaluate(scorers, pairs, base, args.mode, args.filtered, args.keep_source)

    flags = "".join([", filtered" if args.filtered else "", ", source kept" if args.keep_source else ""])
    print(f"{args.split}: {len(pairs)} pairs, {len(base)} base candidates, "
          f"{len(eval_pairs)} pairs evaluated ({args.mode}{flags})")
    print(f"{'model':<18}{'MRR':>7}{'Hit@1':>7}{'Hit@3':>7}{'Hit@10':>8}{'cands':>7}{'cover%':>8}")
    for r in rows:
        print(f"{r['model']:<18}{r['mrr']:>7.2f}{r['hit@1']:>7.2f}{r['hit@3']:>7.2f}{r['hit@10']:>8.2f}"
              f"{r['n_candidates']:>7}{r['coverage']:>8.1f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows({k: r[k] for k in COLS} for r in rows)
    if args.ranks:
        Path(args.ranks).parent.mkdir(parents=True, exist_ok=True)
        with open(args.ranks, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["recipe_id", "source", "target", *per_pair])
            for i, p in enumerate(eval_pairs):
                w.writerow([p.recipe_id, p.source, p.target, *(per_pair[n][i] for n in per_pair)])


if __name__ == "__main__":
    main()
