# Project: ingredient embeddings on recipes

Question (from the poster session): on a small, niche domain, can PPMI, SVD and SGNS with
well-tuned hyperparameters compete with Hugging Face transformer embeddings? The task is
ingredient substitution: given an ingredient in a recipe, rank good substitutes.

| Who | Part |
|---|---|
| Alejandro | data (this folder), SGNS with its hyperparameter search, Scoolinary test set |
| Stanislav | PPMI and SVD with their hyperparameter search |
| Grégoire | transformer embeddings, Recipe1MSubs evaluation (MRR, Hit@k) |

## Data and licences

```bash
OPENRECIPES_GZ=/path/to/20170107-061401-recipeitems.json.gz ./download_recipes.sh
python build_recipe_corpus.py --data-dir data     # ~10 s
python -m pytest tests/
```

- **Open Recipes**, snapshot 2017-01-07, 173,278 recipes, CC BY 3.0 (credit Fictive Kin in the
  report). This is the **training corpus**. The original bucket now answers 403, so the script
  takes a local copy through `OPENRECIPES_GZ` and checks its sha256; ask Alejandro for the file.
- **Recipe1MSubs** (GISMo, Fair, CC-BY-NC): ingredient vocabulary and substitution pairs.
  **Evaluation only, never train on it.** The files are pickles; load them only through
  `recipes.vocabulary`, whose unpickler refuses every class except GISMo's `Vocabulary`.
- **Scoolinary recipes** never enter this repository, not even as derived files. Only aggregate
  numbers go into the report.

## Files you get in `data/recipes/`

| File | Content |
|---|---|
| `ingredients.txt` | **one recipe per line**, its canonical ingredients separated by spaces, each ingredient once, in the order of the recipe (110,469 recipes) |
| `vocab.tsv` | every Recipe1MSubs ingredient: canonical name, GISMo cluster id, number of Open Recipes recipes it appears in |
| `coverage.md` | match rate, most common unmatched lines, substitution-pair coverage |

A canonical name is the first surface form of a GISMo cluster (`garlic_cloves` → `garlic_clove`).
Map any Recipe1MSubs word with `load_vocabulary(...).surface[word]`.

Current numbers: 97.7 % of the ingredient lines that have a name are matched (28 % of all lines
are broken scrapes with only an amount, e.g. `1 cup 1 cup`). 5,797 of 6,633 ingredients appear at
least once. Test pairs with both ingredients in ≥ 10 recipes: 90.4 %; in ≥ 50: 76.4 %.
Matched is not the same as correct: the parser's accuracy is measured later on ~200
hand-labelled lines.


## Rules for training and tuning

1. **Tune on the validation split (`val_comments_subs.pkl`), report on test once.** Tuning on
   test gives oracle numbers, like Table 4 of the paper.
2. **Context = the whole recipe.** An ingredient list is a set, not a sentence, so the window
   size and the dynamic window (`win`, `dyn`) lose their meaning: use a window that covers the
   longest recipe, or count all pairs inside a recipe. Do not spend time tuning them.
3. **Do not reuse `repro/corpus.py` or `repro/sgns.py` as they are.** `load_corpus` reads the
   whole file as one token stream and `train_sgns` cuts it into 10,000-token chunks, so contexts
   would leak from one recipe into the next. Read `ingredients.txt` line by line instead.
   `repro/ppmi_svd.py` (PPMI, SVD, `eig`, `w+c`) works on any count matrix and can be reused.
