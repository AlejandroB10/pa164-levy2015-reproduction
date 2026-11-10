"""Build the canonical-ingredient corpus from Open Recipes.

Usage:
    python build_recipe_corpus.py --data-dir data

Writes into <data-dir>/recipes/:
    ingredients.txt  one recipe per line, its canonical ingredients separated by spaces
    vocab.tsv        every Recipe1MSubs ingredient with its cluster id and recipe count
    coverage.md      match rate, most common unmatched lines, substitution-pair coverage
"""
import argparse
from collections import Counter
from pathlib import Path

from recipes.normalize import Normalizer
from recipes.openrecipes import iter_ingredient_lists, recipe_tokens
from recipes.vocabulary import load_substitutions, load_vocabulary

OPENRECIPES = "openrecipes/20170107-061401-recipeitems.json.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    out = args.data_dir / "recipes"
    out.mkdir(parents=True, exist_ok=True)

    vocab = load_vocabulary(args.data_dir / "recipe1msubs/vocab_ingrs.pkl")
    norm = Normalizer(vocab.surface)
    recipe_count, unmatched = Counter(), Counter()
    n_recipes = n_lines = n_matched = n_quantity_only = 0
    with open(out / "ingredients.txt", "w", encoding="utf-8") as f:
        for lines in iter_ingredient_lists(args.data_dir / OPENRECIPES):
            tokens, missed = recipe_tokens(lines, norm)
            named = [line for line in missed if not norm.is_quantity_only(line)]
            n_lines += len(lines)
            n_matched += len(lines) - len(missed)
            n_quantity_only += len(missed) - len(named)
            unmatched.update(line.lower() for line in named)
            if tokens:
                f.write(" ".join(tokens) + "\n")
                recipe_count.update(tokens)
                n_recipes += 1

    with open(out / "vocab.tsv", "w", encoding="utf-8") as f:
        f.write("ingredient\tcluster_id\trecipes\n")
        for name in vocab.canonical:
            f.write(f"{name}\t{vocab.cluster_id[name]}\t{recipe_count[name]}\n")

    report = [
        "# Open Recipes → Recipe1MSubs vocabulary coverage\n",
        f"- Recipes written: {n_recipes:,} (after removing duplicates and recipes with no match)",
        f"- Ingredient lines: {n_lines:,}, of which {n_quantity_only:,} have no ingredient name "
        "(broken scrapes such as `1 cup 1 cup`)",
        f"- Lines with a name that matched or are headers: {n_matched:,} of {n_lines - n_quantity_only:,} "
        f"({n_matched / (n_lines - n_quantity_only):.1%})",
        f"- Ingredients seen at least once: {len(recipe_count):,} of {len(vocab.canonical):,}",
    ]
    for k in (10, 50, 100):
        report.append(f"- Ingredients in ≥ {k} recipes: {sum(c >= k for c in recipe_count.values()):,}")
    report.append("\n## Substitution pairs with both ingredients in ≥ k recipes\n")
    report.append("| split | pairs | k = 10 | k = 50 |\n|---|---|---|---|")
    for split in ("val", "test"):
        pairs = load_substitutions(args.data_dir / f"recipe1msubs/{split}_comments_subs.pkl", vocab)
        cells = [
            f"{sum(recipe_count[a] >= k and recipe_count[b] >= k for a, b in pairs) / len(pairs):.1%}"
            for k in (10, 50)
        ]
        report.append(f"| {split} | {len(pairs):,} | {cells[0]} | {cells[1]} |")
    report.append("\n## Most common ingredients\n")
    report += [f"{i}. {name}: {c:,}" for i, (name, c) in enumerate(recipe_count.most_common(25), 1)]
    report.append("\n## Most common unmatched lines\n")
    report += [f"- `{line}` ×{c}" for line, c in unmatched.most_common(50)]
    (out / "coverage.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
