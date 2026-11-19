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

Where we are now: 97.7 % of the ingredient lines that name an ingredient get matched. Those
lines leave out the broken scrapes that only kept the amount (`1 cup 1 cup`), which make up
28 % of all lines. 5,797 of the
6,633 ingredients show up at least once, and 90.4 % of the test pairs have both ingredients in
10 or more recipes (76.4 % in 50 or more). A match isn't necessarily a correct match, though.
We still have to check the parser by hand on about 200 lines.

## Shared embedding format

Save your vectors with `recipes.embeddings.save_embeddings(path, names, vectors)`. It writes
plain word2vec text with one canonical ingredient per row, which `load_embeddings` or gensim's
`KeyedVectors.load_word2vec_format` can read back. Then one evaluation script works for every
model. It's fine to leave ingredients out: the evaluation counts them as not covered and reports
the coverage.

## Evaluation (same for every model)

Call `recipes.evaluate.evaluate_substitutions(names, vectors, pairs)`, with `pairs` from
`load_substitutions`. Use val while tuning and test once at the very end.

Each row of the pickle is one user comment, and we count every one of them, duplicates
included. That gives 10,729 instances in val (6,865 distinct pairs) and 10,747 in test (6,798).

For each (target, substitute) pair we rank every ingredient that has a vector, apart from the
target itself, by cosine similarity to the target. If the substitute ends up at rank `r`, the
pair scores `1/r`, and Hit@k checks whether `r <= k`. A pair whose target or substitute has no
vector scores 0, and so does one whose vector is all zeros or not finite. When two candidates
are equally close, the tie counts against the model. Report `mrr` (over all pairs, this is the
headline number), `hit@1/3/10`,
`mrr_covered` (covered pairs only) and `coverage`. That way a model can't look better just by
dropping rare ingredients.

One catch: the candidates are the model's own vocabulary, so the thresholds need to match. SGNS
uses `min_count=5`, which leaves 4,337 ingredients. Say which threshold your model uses.

## SGNS

```bash
python tune_sgns.py grid    # 240 configs on val -> results/recipes/sgns_val.csv (~1 h, resumable)
python tune_sgns.py test    # gensim defaults vs val-best on test, 3 seeds, once
```

`recipes/sgns.py` trains gensim's skip-gram with the window set to the longest recipe and
`shrink_windows=False`. The grid covers `negative` 1/5/15, `ns_exponent` (cds) 0.75/1.0,
`sample` 0/1e-5/1e-4/1e-3, `vector_size` 100/300 and `epochs` 5 to 100, and each model gets
scored twice, as W and as W+C. The script sets `PYTHONHASHSEED=0` and gives each config a
single worker. gensim seeds its vectors from Python's string hash, and without a fixed seed two
runs of the same config come out different. Embeddings are saved to
`data/recipes/embeddings/sgns-{vanilla,tuned}.txt`.

Test results, mean ± std over 3 seeds (every row is in `results/recipes/sgns_test.csv`):

| Model | Config | MRR | Hit@1 | Hit@10 | Coverage |
|---|---|---|---|---|---|
| SGNS vanilla | gensim defaults: neg 5, cds 0.75, sub 1e-3, dim 100, 5 epochs | 0.058 ± 0.000 | 0.036 | 0.100 | 94.3 % |
| SGNS tuned | val-best: neg 15, cds 0.75, sub 1e-3, dim 100, 100 epochs, W only | 0.101 ± 0.000 | 0.056 | 0.195 | 94.3 % |

On val, the number of epochs made the biggest difference: going from 5 to 100 took MRR from
0.067 to 0.105. `neg` came second (0.099 with 1, 0.105 with 15). 100 dimensions did better than
300, and `cds` hardly mattered. The surprise was W+C. On text8 it helped, but here it only beat
W in 43 of the 240 configurations, nearly all of them weak ones (5 epochs, or `sample` 1e-5), and
its best run reached 0.074 against 0.105 for W.

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
