"""Draw ingredient lines from Open Recipes to check the normaliser by hand.

Usage:
    python sample_parser_lines.py --data-dir data --out results/recipes/parser_sample.tsv
    python sample_parser_lines.py --score results/recipes/parser_sample.tsv

The sample is drawn with a fixed seed from the deduplicated recipes, leaving out section headers
and the broken lines that only kept an amount, since those never name an ingredient. Each row
shows the line and what the normaliser returned. Whoever labels it fills `correct` with y or n
and, when it is n, writes the right canonical ingredient in `expected` (empty if the line names
no ingredient in the vocabulary), with an optional note. `--score` then prints the accuracy.
"""
import argparse
import csv
import random
from pathlib import Path

from recipes.normalize import Normalizer
from recipes.openrecipes import iter_ingredient_lists
from recipes.vocabulary import load_vocabulary

OPENRECIPES = "openrecipes/20170107-061401-recipeitems.json.gz"
SEED = 164
COLUMNS = ["line", "parsed", "correct", "expected", "notes"]


def draw(data_dir, size):
    norm = Normalizer(load_vocabulary(data_dir / "recipe1msubs/vocab_ingrs.pkl").surface)
    lines = [line for recipe in iter_ingredient_lists(data_dir / OPENRECIPES) for line in recipe
             if not norm.is_header(line) and not norm.is_quantity_only(line)]
    picked = random.Random(SEED).sample(lines, size)
    return [{"line": line, "parsed": norm.normalize(line) or "", "correct": "", "expected": "", "notes": ""}
            for line in picked]


def score(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["correct"] in ("y", "n")]
    if not rows:
        raise SystemExit(f"{path}: no labelled rows yet")
    right = sum(r["correct"] == "y" for r in rows)
    matched = [r for r in rows if r["parsed"]]
    print(f"labelled {len(rows)}; accuracy {right / len(rows):.1%}; "
          f"precision of matched lines {sum(r['correct'] == 'y' for r in matched) / max(len(matched), 1):.1%} "
          f"({len(matched)} matched)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/recipes/parser_sample.tsv"))
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--score", type=Path, help="print the accuracy of a labelled sample")
    args = parser.parse_args()
    if args.score:
        return score(args.score)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; it may hold labels, so it is not overwritten")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(draw(args.data_dir, args.size))
    print(f"{args.size} lines -> {args.out}")


if __name__ == "__main__":
    main()
