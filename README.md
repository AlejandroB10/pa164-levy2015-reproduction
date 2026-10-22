# Reduced-scale reproduction: Levy, Goldberg & Dagan (TACL 2015) on text8

Claim under test (§5.1): *"Selecting the right hyperparameter settings often has more
impact than choosing the most suitable algorithm."* For PPMI, SVD and SGNS we compare the
paper's "vanilla" setting (Table 2) with the rules of thumb from §7, on one small corpus
and one shared vocabulary.

## Quick start

You need: Linux or macOS (on Windows, use WSL), [Miniforge or conda](https://github.com/conda-forge/miniforge),
`curl`, `unzip`, about 1.5 GB of free disk and 4 GB of RAM (peak use is 2.8 GB). No GPU is needed.

```bash
git clone <this repo's URL>
cd pa164-levy2015-reproduction

# 1. Environment (once)
conda env create -f environment.yml
conda activate pa164-levy2015

# 2. Data (once, ~130 MB into ./data, which git ignores)
./download_data.sh

# 3. Quick check (~1 min): must print "3CosAdd 0.440" and "OK"
python sanity_check.py --data-dir data

# 4. Full experiment (~10-30 min, depending on your CPU)
python run_all.py --data-dir data --out results-mine/ --workers 8

# 5. Compare your numbers with ours
python compare_results.py results-mine/results.csv
```

- `--workers`: set it to your number of CPU cores (`nproc` on Linux).
- Step 4 writes to `results-mine/` so it does not overwrite the reference results in `results/`.
- Step 5 prints `OK` if PPMI and SVD match ours within 0.01 and SGNS within 0.04. PPMI and SVD
  are deterministic and should match exactly. SGNS is not: multithreaded gensim gives slightly
  different vectors with a different number of workers (up to 0.03 in our test, on the small sets).
- If step 4 is interrupted, just run it again: finished pieces are cached in `data/cache/` and reused.

`run_all.py` caches the co-occurrence matrix, SVD factors and SGNS vectors under
`<data-dir>/cache/` (about 850 MB, ignored by git). Measured cold runs on a 20-core CPU-only
laptop: 9 min 05 s with `--workers 8` from a fresh clone, and 25 min 29 s in an earlier run with
`--workers 20`. A re-run from cache takes about 5 min.

Tested end to end from a fresh copy of this repo (download, sanity check, full run, comparison):
the sanity check printed 0.440, PPMI and SVD matched the reference exactly, and SGNS stayed within 0.03.

Outputs: `results.csv` (one row per configuration and metric, with `n_evaluated` and `n_total`
for coverage) and `poster_chart.png`. The `build_seconds` column holds the build time of that run:
the committed CSV comes from a cached re-run, so SVD and SGNS show ~0 s there.
Cold timings: SVD 170 s (cds=1) and 129 s (cds=0.75); SGNS 111 s (neg=1), 218 s (neg=5), 539 s (neg=15).

## Code

| File | Role |
|---|---|
| `repro/corpus.py` | read text8, build the frequency-ordered vocabulary, encode to ids |
| `repro/counts.py` | symmetric word-context counts in a fixed window |
| `repro/ppmi_svd.py` | shifted PPMI with context-distribution smoothing; truncated SVD; `eig` and `w+c` |
| `repro/sgns.py` | gensim skip-gram with negative sampling, aligned to the shared vocabulary |
| `repro/evaluation.py` | Spearman ρ on similarity sets; Google analogies with 3CosAdd and 3CosMul |
| `repro/plotting.py` | poster chart |
| `run_all.py` | the whole experiment |
| `sanity_check.py` | rebuilds an earlier prototype setting to validate the pipeline |
| `compare_results.py` | compares a fresh run with the committed `results/results.csv` |
| `download_data.sh` | downloads text8 and the evaluation sets into `./data` |

The formulas were re-implemented from the paper; the original `hyperwords` code was only read
as a reference (it is not executed and no code was copied).

## Setup

- **Corpus:** text8, 17,005,207 tokens.
- **Vocabulary:** words with count ≥ 5, i.e. 71,290 words. Out-of-vocabulary tokens are deleted
  before counting and training, leaving 16,718,844 tokens. Every method uses exactly the same token
  stream and vocabulary.
- **Window:** `win=2`, uniform weighting: every context within 2 words counts 1. `dyn=none`
  (gensim `shrink_windows=False`), `sub=none` (gensim `sample=0`).
- **PPMI:** PMI_α(w,c) = log(#(w,c) / #(w) / P_α(c)), with P_α(c) = #(c)^α / Σ #(c')^α (`cds` = α).
  Shifted PPMI = max(PMI_α − log k, 0) (`neg` = k).
- **SVD:** 300 dims, ARPACK (`scipy.sparse.linalg.svds`) on the PPMI matrix with neg=1.
  Word vectors = U·Σ^eig. `w+c` adds V·Σ^eig before normalisation.
- **SGNS:** gensim `Word2Vec(sg=1, hs=0, window=2, vector_size=300, epochs=5, seed=1, workers=20)`.
  `neg` = `negative`, `cds` = `ns_exponent`. `w+c` = input vectors + `syn1neg` output vectors,
  added raw and then L2-normalised (as hyperwords does).
- **Normalisation:** all vectors are row-L2-normalised before evaluation.
- **Similarity:** Spearman ρ between cosine and human scores. Pairs with an out-of-vocabulary word
  are dropped (hyperwords instead scores them 0). Coverage: WS-Sim 202/203, WS-Rel 251/252,
  MEN 2998/3000, MTurk 284/287, Rare Words 951/2034, SimLex 992/999.
- **Analogy (Google):** 3CosAdd and 3CosMul. The three question words are excluded from the
  candidates. For 3CosMul, ε = 0.001 (paper §4.3). Dense cosines are shifted to [0,1] via (x+1)/2;
  sparse PPMI cosines are already ≥ 0 and are used as is (same as hyperwords). Accuracy is computed
  on the questions whose four words are in the vocabulary. There are two variants:
  - **`*_30k` (primary):** questions and candidate answers limited to the 30,000 most frequent
    words. 12,268 / 19,544 questions (62.8 %).
  - **full (secondary):** all 71,290 words are candidates. 17,827 / 19,544 questions (91.2 %).

### Why the analogy evaluation is restricted to 30k words

With the full 71k vocabulary, poorly estimated rare words win the argmax. A diagnostic on vanilla
PPMI (first 2,000 covered questions) found:
- the median frequency rank of the predicted word is 54,381, while that of the gold word is 2,738;
- 533 of 2,000 3CosMul predictions have cos(prediction, a) = 0, so the ε = 0.001 denominator
  multiplies their score by 1000.

Restricting the candidates moves vanilla PPMI 3CosAdd from 0.220 (full) to 0.323 (top 30k) and
0.462 (top 10k). The paper does not face this problem, because it keeps only words seen ≥ 100 times
in 1.5B tokens. Both variants are in the CSV.

## Sanity check

| Run | 3CosAdd | Questions |
|---|---|---|
| Earlier standalone prototype (not in this repo), evaluated with gensim | 44.0 % | 12,268 |
| This package, same setting (`sanity_check.py`), own evaluator | **44.0 %** (3CosMul 46.6 %) | 12,268 |

The setting is: top-30k vocabulary, window 5 with 1/d weights, cds=0.75, 300-dim SVD, eig=0.5.
The two numbers match, which validates the counting, PPMI, SVD and analogy code.

## Results (text8, win=2, 300 dims)

The "role" column marks the paper's vanilla setting and the setting picked *a priori* from the §7
rules of thumb. The poster chart compares these two, plus the recommended setting with w+c for
SVD and SGNS (§7 suggests trying w+c; it does not apply to PPMI). All three are fixed in advance,
so no test-set selection is involved.

| Config | role | WS-Sim | WS-Rel | MEN | MTurk | RW | SimLex | Add@30k | Mul@30k | Add (full) | Mul (full) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PPMI vanilla (cds=1, neg=1) | vanilla | .587 | .545 | .548 | .570 | .205 | .252 | .323 | .127 | .220 | .007 |
| PPMI cds=0.75 | recommended | .614 | .569 | .567 | .582 | .250 | .275 | .345 | .337 | .244 | .081 |
| PPMI cds=0.75 neg=5 | variant | .638 | .594 | .606 | .582 | .286 | .283 | .230 | .070 | .137 | .021 |
| SVD vanilla (cds=1, eig=0) | vanilla | .695 | .630 | .645 | .569 | .355 | .276 | .354 | .370 | .268 | .282 |
| SVD cds=0.75 eig=0 | variant | .675 | .628 | .608 | .599 | .284 | .246 | .317 | .335 | .240 | .252 |
| SVD cds=0.75 eig=0.5 | recommended | .675 | .628 | .599 | .618 | .277 | .234 | .336 | .360 | .255 | .274 |
| SVD cds=0.75 eig=0.5 w+c | variant | .679 | .653 | .639 | .612 | .299 | .246 | .333 | .348 | .263 | .277 |
| SVD cds=0.75 eig=1 | variant | .611 | .543 | .524 | .583 | .202 | .212 | .341 | .362 | .261 | .277 |
| SGNS vanilla (neg=1, cds=1) | vanilla | .638 | .575 | .478 | .543 | .325 | .276 | .246 | .233 | .178 | .165 |
| SGNS neg=5 cds=0.75 | variant | .663 | .543 | .485 | .500 | .373 | .321 | .309 | .311 | .222 | .227 |
| SGNS neg=15 cds=0.75 | recommended | .691 | .572 | .491 | .537 | .380 | .334 | .318 | .322 | .225 | .233 |
| SGNS neg=15 cds=0.75 w+c | variant | .718 | .672 | .603 | .607 | .346 | .301 | .387 | .364 | .285 | .275 |

The paper's numbers are given for context only and are **not comparable**: they use 1.5B tokens,
a count cutoff of 100 and 500 dims. Paper Table 5 (win=2, tuned): WS-Sim SGNS .789,
SVD .772, PPMI .732; MSR Add/Mul SGNS .617/.644 (MSR was not evaluated here).

## Takeaways

- **The claim is only partly reproduced at 17M tokens.** On WS-Sim, the spread between vanilla
  algorithms (.587 to .695) is larger than most single-tweak effects. Tuning, however, closes or
  reverses the gap for SGNS: tuned SGNS (.691, .718 with w+c) matches or beats vanilla SVD (.695).
  It does not do the same for PPMI.
- **cds=0.75 helps PPMI on all six similarity sets and on analogies, but not SVD.** For SVD on this
  corpus it lowers MEN (.645 → .608) and Rare Words (.355 → .284). The §7 claim "always use cds"
  does not hold here for SVD.
- **eig=1 is a clear penalty on similarity, as the paper warns:** −6 to −9 points against eig=0.5
  on WS-Sim, WS-Rel, MEN and RW. On analogies it makes no difference here.
- **w+c is the largest single gain for SGNS:** WS-Rel +.10, MEN +.11, Add@30k +.07, at zero
  training cost. Many negatives help SGNS moderately (vanilla → neg=15: WS-Sim +.05, SimLex +.06,
  Add@30k +.07). Shifted PPMI (neg=5) helps similarity but hurts analogies.
- **The evaluation protocol matters as much as the hyperparameters.** With rare words as
  candidates, PPMI 3CosMul falls to 0.007. This is an artifact of sparse vectors combined with
  ε = 0.001, not a property of the representation (see above).

## Deviations from the paper

1. Corpus: text8 (17M tokens) instead of Wikipedia (1.5B tokens). Absolute numbers are much lower.
2. Vocabulary: count ≥ 5 (71k words) instead of ≥ 100 (189k words). Analogies are evaluated
   primarily on the top 30k.
3. Dimensions: 300 instead of 500, to keep SVD time and memory bounded.
4. Only win=2; `dyn` and `sub` were not varied (both off), and GloVe was not run.
   The win=5 column was skipped to stay within the time budget.
5. Vanilla SVD uses **eig = 0**, following the Table 2 (and Table 3) caption. §6.1 says instead that
   eig = 0.5 was "the default setting for SVD in the vanilla scenario". The paper is inconsistent
   here; eig=0.5 appears in the table as a separate row.
6. 3CosMul uses ε = 0.001 as in the paper; the hyperwords code uses 0.01.
7. No hyperparameter search or cross-validation. A fixed vanilla setting and a fixed a-priori
   setting were compared (not Table 5's 2-fold CV).
8. SGNS: 5 epochs (the paper uses 1 pass over 1.5B tokens). The stream is cut into 10,000-token
   "sentences" for gensim, so a few contexts at the cuts are lost. Multithreaded gensim is not
   bit-reproducible: the earlier prototype saw up to ±0.7 points between runs with the same seed.
9. Out-of-vocabulary similarity pairs are dropped instead of scored 0. Coverage is listed above.
10. MSR analogies were not evaluated (dataset not available locally).
