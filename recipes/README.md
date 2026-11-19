# Project: ingredient embeddings on recipes

At the poster session the teacher asked whether count models and SGNS, once their
hyperparameters are tuned, can keep up with Hugging Face transformer embeddings on a small,
niche domain. We test that on ingredient substitution: given an ingredient in a recipe, rank the
ingredients that could replace it.

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

Open Recipes is the training corpus: the 2017-01-07 snapshot, 173,278 recipes, CC BY 3.0
(credit Fictive Kin in the report). The original bucket returns 403 these days, so the script
takes a local copy through `OPENRECIPES_GZ` and checks its sha256. Ask Alejandro for the file.

Recipe1MSubs (GISMo, by Fair, CC-BY-NC) gives us the ingredient vocabulary and the substitution
pairs. We only evaluate on it and never train on it. Its files are pickles, so load them through
`recipes.vocabulary` and nothing else: that unpickler refuses every class except GISMo's
`Vocabulary`.

The Scoolinary recipes never go into this repository, not even as derived files. The report
only shows aggregate numbers for them.

## Files you get in `data/recipes/`

| File | Content |
|---|---|
| `ingredients.txt` | one recipe per line: its canonical ingredients, separated by spaces, each once, in recipe order (110,469 recipes) |
| `vocab.tsv` | every Recipe1MSubs ingredient with its canonical name, GISMo cluster id and the number of Open Recipes recipes it appears in |
| `coverage.md` | match rate, most common unmatched lines, substitution-pair coverage |

A canonical name is the first surface form of a GISMo cluster, so `garlic_cloves` becomes
`garlic_clove`. To map any Recipe1MSubs word, use `load_vocabulary(...).surface[word]`.

Where we are now: 97.7 % of the ingredient lines that name an ingredient get matched. Another
28 % of all lines are broken scrapes that only kept the amount (`1 cup 1 cup`). 5,797 of the
6,633 ingredients show up at least once, and 90.4 % of the test pairs have both ingredients in
10 or more recipes (76.4 % in 50 or more). A match isn't necessarily a correct match, though.
We still have to check the parser by hand on about 200 lines.

## Rules for training and tuning

1. Tune on the validation split (`val_comments_subs.pkl`) and report on test once. If you tune
   on test you get oracle numbers, like Table 4 of the paper.
2. The context is the whole recipe. An ingredient list is a set, not a sentence, so window size
   and the dynamic window (`win`, `dyn`) mean nothing here. Use a window that covers the longest
   recipe, or count all pairs inside a recipe, and don't spend time tuning either of them.
3. Don't reuse `repro/corpus.py` or `repro/sgns.py` as they are. `load_corpus` reads the whole
   file as one token stream and `train_sgns` cuts it into 10,000-token chunks, so contexts would
   leak from one recipe into the next. Read `ingredients.txt` line by line instead.
   `repro/ppmi_svd.py` (PPMI, SVD, `eig`, `w+c`) works on any count matrix, so that one you can
   reuse.
