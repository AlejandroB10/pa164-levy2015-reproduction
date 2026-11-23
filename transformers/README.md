# Transformers and evaluation (Grégoire's part)

This folder holds the transformer embeddings and the Recipe1MSubs evaluation for the recipe
project. Run everything from inside `transformers/`. The data comes from `../download_recipes.sh`,
which puts Recipe1MSubs in `../data/recipe1msubs`.

| What | File |
|---|---|
| Transformer vectors in the shared format, scored with `recipes/evaluate.py` | `export_shared.py` → `../results/recipes/transformers_test_val.csv` |
| BGE and averaged BERT, context-free | `embed_hf.py`, tuned on val with `tune_hf.py` → `../results/recipes/transformers_val_grid.csv` |
| BGE and BERT averaged over recipe ingredient lists | `embed_context.py` |
| Evaluation with a shared candidate set, GISMo baselines, back-off | `run_eval.py`, `subeval/` |
| Popularity prior, ensembles, back-off hybrid | `tune_combo.py`, `subeval/combine.py` |
| Bootstrap CIs, seen/unseen pairs, neighbours | `analyze.py` |
| **Stand-in** PPMI/SVD/SGNS on Recipe1M train lists, only to test the pipeline | `classic_recipes.py` |

`run_eval.py --mode own` follows the same protocol as `recipes/evaluate.py`: the model's own
vocabulary is the candidate set, the source is excluded, ties count against the model, and an
uncovered pair scores 0. On the same vectors the two give the same MRR. `--mode common`
additionally gives every model the same candidates.

The PPMI/SVD/SGNS numbers below come from `classic_recipes.py`. These are quick **stand-ins**
trained on the 31k Recipe1M train ingredient lists, not the team's Open Recipes models. They will
be replaced by Alejandro's SGNS and Stanislav's PPMI/SVD.

Commands below use paths relative to `transformers/`. Heavy outputs (`vectors/`, `results/`,
`shared/`) are git-ignored.

---

Ingredient-substitution evaluation for the PA164 recipe project. For each pair
(source → target) in Recipe1MSubs, the model ranks every candidate ingredient by how good a
substitute for the source it is. The target's rank gives **MRR** and **Hit@1/3/10**, in %,
as in the GISMo paper (Fatemi et al., 2023, [arXiv:2302.07960](https://arxiv.org/abs/2302.07960)).

## Data

Put these four files from `https://dl.fbaipublicfiles.com/gismo/` in one folder, for example
`../data/recipe1msubs/` (about 6 MB in total):

```
train_comments_subs.pkl   val_comments_subs.pkl   test_comments_subs.pkl   vocab_ingrs.pkl
```

The pickles are read with a restricted unpickler. It only accepts plain containers and the
vocabulary class, so a tampered file cannot run code.

GISMo's candidate set is the list of FlavorGraph ingredient nodes, `nodes_191120.csv`
(343 KB, https://github.com/lamypark/FlavorGraph/tree/master/input). Put it in
`../data/flavorgraph/` and pass it with `--candidates`. It has 6,653 ingredients: the 6,632
canonical names of the vocabulary, plus 21 that are never a target, so the numbers are the same
either way.

## Shared vector format (ingredient → vector)

Each method exports one file:

- **`.txt`**: word2vec text format, one line per ingredient: `olive_oil 0.12 -0.03 ...`,
  with an optional `n_words dim` header line. Names must not contain spaces: use `_`.
- **`.npz`**: arrays `words` (strings) and `vectors` (n × d), e.g.
  `np.savez("sbert.npz", words=np.array(names), vectors=mat)`.

Names are normalized on load: lower case, with spaces and hyphens turned into `_`. The
evaluation looks names up by the canonical Recipe1M name (`butter` for `unsalted_butter`,
`oil` for `vegetable_oil`). Name each vector by that canonical name, or by a synonym that maps
to it. A vector that is all zeros counts as missing.

## Run

Use Python 3.12 or later, with numpy.

```bash
# tuning: validation split only
python run_eval.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
    --split val --model sgns=vectors/sgns_w5_neg15.txt

# final table: test split, every method on the same candidates and pairs
python run_eval.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
    --split test --model sgns=vectors/sgns.txt --model ppmi_svd=vectors/svd.npz \
    --model sbert=vectors/sbert.npz --model bert_avg=vectors/bert_avg.npz \
    --baselines freq pair-freq pmi gismo-ltfreq --out results/test.csv --ranks results/test_ranks.csv

# sanity check: reproduces GISMo's Table 1 baselines
python run_eval.py --data-dir ../data/recipe1msubs --candidates ../data/flavorgraph/nodes_191120.csv \
    --split test --keep-source --baselines gismo-freq gismo-lt gismo-ltfreq

python -m unittest discover -s tests
```

## Protocol choices (these go in the report)

| Choice | What the script does | Why |
|---|---|---|
| Candidate set | By default, every canonical ingredient in `vocab_ingrs.pkl`. With `--candidates`, the FlavorGraph ingredient nodes, as in GISMo | Ranking against the full ingredient list, as GISMo does |
| `--mode common` (default) | Candidates are limited to the ingredients that **every** model knows. Pairs are kept only if both ingredients are in that set | All methods face the same task, so the numbers can be compared |
| `--mode own` | Each model ranks its own vocabulary. A pair it cannot score counts as a miss (reciprocal rank 0) | Low coverage is penalized instead of hidden. Report it alongside the coverage column |
| Source ingredient | Not a candidate. `--keep-source` keeps it, as GISMo's code does | A cosine scorer would always rank the source first. The paper ranks $D_{ing}\setminus\{s\}$ |
| Ties | Counted against the target (`>=`, as in GISMo) | A constant scorer gets the worst rank, not a lucky one |
| Filtering | Off by default. `--filtered` removes the other gold targets of the same (recipe, source) | The paper's text says it accepts "all valid target ingredients as correct answers", but its Table 1 numbers can only be reproduced unfiltered (`filter=False` in its README). Filtering adds about 1 MRR point |
| Recipe context | Not used | Our embeddings are context-free. Compare with GISMo's context-free numbers |

Baselines (context-free, fitted on the **train** split only):

- `freq`: how often each ingredient is a substitution target (popularity).
- `pair-freq`: count of (source → candidate) in train, with ties broken by `freq`. This is
  GISMo's LT+Freq with a tie-breaker, and the strongest baseline.
- `gismo-freq`, `gismo-lt`, `gismo-ltfreq`: GISMo's baselines, copied exactly from
  `gismo/models.py`. Note that their `Freq` sums the lookup table over dimension 1, so it
  counts how often a candidate is a **source**, although the paper says "target". Use these
  to quote the paper's baselines; use `freq` and `pair-freq` in our own table.
- `pmi`: PMI between two ingredients that appear in the same recipe. It uses the recipes of
  the train split, or the file given with `--pmi-recipes` (one recipe per line, ingredients
  separated by tabs). Ingredients that appear together are complements more than
  substitutes, which makes this a useful weak baseline.

## Reference numbers (FlavorGraph candidates, 6,653)

The split sizes match the paper exactly: 49,044 / 10,729 / 10,747 pairs.

**Reproduction of GISMo's Table 1** (test, `--keep-source`, unfiltered). MRR / H@1 / H@3 / H@10:

| Baseline | Ours | Paper |
|---|---|---|
| gismo-freq | 3.80 / 1.61 / 2.77 / 6.41 | Freq 3.81 / 1.61 / 2.77 / 6.41 |
| gismo-lt | 2.55 / 0.15 / 0.89 / 5.96 | LT 2.53 / 0.15 / 0.88 / 5.80 |
| gismo-ltfreq | 27.62 / 18.00 / 32.19 / 47.71 | LT+Freq 27.66 / 18.10 / 32.26 / 47.50 |

The match is within 0.1 MRR, so the evaluation computes the same thing as GISMo's.

**Our protocol** (default flags: source excluded, unfiltered):

| Baseline | Val | Test |
|---|---|---|
| freq | 5.71 / 2.56 / 4.92 / 11.81 | 5.68 / 2.45 / 4.93 / 11.98 |
| pair-freq | 29.35 / 19.41 / 33.84 / 50.13 | 29.56 / 19.10 / 34.28 / 51.16 |
| gismo-ltfreq | 27.47 / 18.26 / 31.64 / 47.05 | 27.62 / 18.00 / 32.19 / 47.71 |
| pmi (train recipes) | 0.29 / 0.01 / 0.07 / 0.21 | 0.28 / 0.01 / 0.07 / 0.25 |

Why our `freq` is higher than the paper's: it counts targets, as the paper describes, and it
excludes the source. Under GISMo's code, `butter` is the most frequent source in train (1,740
pairs), so their `Freq` ranks butter first for every pair, even when butter is the source
itself. Water (1,732) and sugar (1,156) come next, which are poor substitutes. `pair-freq`
gains 2 points over `gismo-ltfreq` because of its tie-breaker.

For comparison, the paper's other test MRRs: Random 0.15, Food2Vec 9.65, FoodBERT 9.32,
R-FoodBERT 10.09, and GISMo, which uses the recipe context, 31.51. PMI over co-occurring
ingredients is barely better than random: ingredients that appear together are complements,
not substitutes.

## Hugging Face embeddings (transformers part)

`embed_hf.py` encodes every FlavorGraph ingredient name on its own (`olive_oil` →
"olive oil"), so the vector does not depend on the recipe, like SGNS / PPMI / SVD. It writes
the shared `.npz` format. It runs on CPU, offline from the Hugging Face cache, in about 90 s
per model.

- **Sentence-embedding model**: `BAAI/bge-base-en-v1.5`, [CLS] pooling (its own setting).
- **Averaged BERT**: `bert-base-cased`, mean of the word-piece vectors of the name only
  (template words, [CLS] and [SEP] are left out).

`tune_hf.py` picks the settings on **val only**. The grid is 2 models × 3 templates
(`{}`, `{} (an ingredient)`, `a recipe with {}`) × 2 poolings (cls, mean) × 10 layer sets,
120 settings in total. Each (model, template) pair needs one forward pass, about 15 min for
the whole grid. Results: `results/val_hf_grid.csv`.

```bash
python embed_hf.py --model BAAI/bge-base-en-v1.5 --pooling cls --layers 1 2 3 4 5 6 7 8 9 10 11 12 \
    --template "{} (an ingredient)" --candidates ../data/flavorgraph/nodes_191120.csv --out vectors/bge.npz
python embed_hf.py --model bert-base-cased --pooling mean --layers -4 -3 -2 -1 \
    --template "{} (an ingredient)" --candidates ../data/flavorgraph/nodes_191120.csv --out vectors/bert_avg.npz
```

Results (MRR / H@1 / H@3 / H@10). "Vanilla" means no template: [CLS] of the last layer for
BGE, mean of the last layer for BERT.

| Model | Val | Test |
|---|---|---|
| bge vanilla | 3.70 / 0.73 / 2.74 / 8.61 | 3.57 / 0.85 / 2.59 / 7.73 |
| bge tuned (template, cls, all layers) | 4.39 / 1.13 / 3.57 / 10.54 | 4.08 / 1.03 / 3.51 / 9.55 |
| bert_avg vanilla | 3.35 / 1.32 / 3.42 / 7.76 | 3.16 / 1.24 / 3.08 / 7.33 |
| bert_avg tuned (template, mean, last 4) | 5.67 / 3.41 / 5.22 / 11.06 | 5.29 / 3.10 / 4.86 / 10.30 |
| freq (baseline) | 5.71 / 2.56 / 4.92 / 11.81 | 5.68 / 2.45 / 4.93 / 11.98 |

What the grid and the neighbours show:

- **Tuning matters more than the model.** Picking the template, pooling and layers is worth
  +0.7 MRR for BGE and +2.1 for BERT, while the gap between the two vanilla models is 0.4.
  This is the same lesson as Levy et al.
- **The sentence model loses to plain BERT.** BGE's neighbours are surface variants: for
  `butter` it returns `apple_butter`, `honey_butter`, `garlic_butter`. Retrieval training
  rewards matching names, not interchangeable ingredients. Averaged BERT groups by category
  instead: oils with oils, creams with creams.
- **The template "(an ingredient)" helps both models.** It pushes the name towards its food
  sense. "a recipe with {}" helps less.
- **Context-free transformers do not beat popularity.** Tuned BERT is level with `freq` at
  Hit@1, higher on val (3.41 vs 2.56), but below it on MRR and Hit@10. All of them are far
  from the paper's Food2Vec and FoodBERT (about 9.5), which were adapted to recipe text. Our
  models were not.

`--ranks` writes one row per pair with every model's rank (0 = not covered). Use it for error
analysis and for significance tests between models, such as a paired bootstrap on the
reciprocal ranks (`analyze.py`, below).

## Provisional classic models (`classic_recipes.py`)

These models stand in until the team's Open Recipes models are ready. They use the same
methods and knobs as Levy et al., trained on the ingredient lists of the **31,097 train
recipes** of Recipe1MSubs (5,484 ingredients). The context is the whole recipe.

```bash
python classic_recipes.py --data-dir ../data/recipe1msubs --grid --out results/val_classic_grid.csv
python classic_recipes.py --data-dir ../data/recipe1msubs --method sgns --cds 1 --neg 15 --dim 100 --epochs 30 --out-vectors vectors/sgns_recipes.npz
```

The grid covers 84 settings: PPMI (cds × neg), SVD (cds × neg × dim × eig × w+c) and SGNS
(dim × neg × cds × epochs × w+c). The best settings on val were:

- PPMI: cds=0.75.
- SVD: dim=100, eig=1, cds=1.
- SGNS: neg=15, dim=100, cds=1, 30 epochs, no w+c.

On recipes, eig=1 and cds=1 win, which is the opposite of Levy et al.'s rules of thumb.

## Joint evaluation, test (`results/test_all.csv`, common mode, 5,484 candidates)

| Model | Vanilla MRR | Tuned MRR | Tuned H@1 | Tuned H@10 |
|---|---|---|---|---|
| PPMI | 3.89 | 4.12 | 1.45 | 8.93 |
| SVD | 2.54 | 5.20 | 2.33 | 9.43 |
| SGNS | 4.74 | **6.27** | 3.63 | 11.17 |
| BGE | 3.91 | 4.47 | 1.15 | 10.67 |
| Averaged BERT | 3.37 | 5.63 | 3.25 | 11.03 |
| Contextual BERT (`embed_context.py`, layer 8) | – | 5.02 | 2.32 | 10.36 |
| Contextual BGE (`embed_context.py`, layer 12) | – | 4.92 | 1.22 | **13.05** |
| freq | | 5.79 | 2.49 | 12.24 |
| pair-freq | | 30.21 | 19.56 | 52.26 |
| pmi | | 0.29 | 0.01 | 0.26 |

Vanilla means the default settings. For PPMI that is cds=1. For SVD it is eig=0 and 300
dimensions. For SGNS it is word2vec's defaults (neg=5, cds=0.75, 10 epochs). For the
transformers it means no template.

## Combinations (`tune_combo.py`, `subeval/combine.py`)

For one source, each component's scores are standardized over the candidates and then
summed with weights. `logfreq` is log(1 + how often the candidate is a target in train), a
popularity prior. `pair-backoff` puts the seen (source, target) pairs first, so the other
components only order the rest. Weights are chosen on val, and test is scored once.

| Setting (best on val) | Test MRR | H@1 | H@10 |
|---|---|---|---|
| bge + 0.5·logfreq | 10.84 | 5.59 | 22.18 |
| sgns + 1·logfreq | 13.83 | 6.43 | 29.73 |
| **sgns + 0.75·bge + 1·logfreq** | **15.22** | 7.34 | 32.55 |
| pair-backoff + logfreq (= pair-freq) | 30.21 | 19.56 | 52.26 |
| **pair-backoff + sgns + 1·bge + 1·logfreq** | **30.98** | 19.94 | 54.03 |

For comparison, the GISMo paper's test MRRs are Food2Vec 9.65, FoodBERT 9.32 and GISMo
31.51. Their language-model baselines use no popularity prior. The prior and pair-freq use
the train substitutions, which is a light supervised signal. The embeddings alone use none.

## Analysis (`analyze.py` → `results/analysis_test.md`)

The analysis reports bootstrap confidence intervals, paired tests between every pair of
models, results split into seen and unseen pairs and by source frequency, and top-5
neighbours. The main findings:

- **35% of test pairs are unseen in train.** On those pairs pair-freq gets 0.82 MRR and
  sgns + bge + logfreq gets 4.17. The +0.77 of the back-off hybrid over pair-freq
  (p < 0.001) comes from unseen pairs and from rare sources.
- **The neighbours show four kinds of similarity.**
  - SGNS finds substitutes, because they share contexts: margarine for butter, stock for broth.
  - SVD with eig=1 finds complements, because they appear in the same recipes: flour and egg for butter.
  - BGE finds names that look alike: apple butter.
  - BERT finds the same food category: milk and cream.
- **Recipe context hurts BERT but helps BGE.** Each vector is averaged over up to 30 recipes
  per ingredient. Contextual BERT scores 5.02, against 5.63 when it encodes the name alone
  with a template: the words next to an ingredient in a list are its complements. Contextual
  BGE improves on BGE alone (4.92 against 4.47) and has the best Hit@10 of every single model
  (13.05). In combinations, though, plain BGE does better (sgns + bge + prior 15.22, against
  14.13 with ctx_bge). Encoding takes 25 to 40 min per model on CPU. The `*_layers.npz` files
  keep every layer, about 265 MB each.
