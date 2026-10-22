"""Mini-ablation of Levy, Goldberg & Dagan (TACL 2015) on text8.

For PPMI, SVD and SGNS, compare the paper's "vanilla" setting with its recommended
tweaks (cds=0.75, symmetric eig, many negatives, w+c), all on one shared vocabulary
and one window size, and evaluate on 6 word-similarity sets and Google analogies.

Usage:
    python run_all.py --data-dir PATH/TO/ML-NLP-data --out results/
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from repro.corpus import load_corpus
from repro.counts import cooccurrence
from repro.evaluation import (
    SIMILARITY_SETS, Space, evaluate_analogy, evaluate_similarity, load_analogies, load_similarity,
)
from repro.plotting import poster_chart
from repro.ppmi_svd import ppmi, svd_factors, svd_vectors
from repro.sgns import train_sgns

# (config name, role, params). role marks the paper's vanilla setting and the setting
# chosen a priori from the paper's Sec. 7 rules of thumb (used for the poster chart).
PPMI_CONFIGS = [
    ("PPMI vanilla (cds=1, neg=1)", "vanilla", dict(cds=1.0, neg=1)),
    ("PPMI cds=0.75", "recommended", dict(cds=0.75, neg=1)),
    ("PPMI cds=0.75 neg=5", "variant", dict(cds=0.75, neg=5)),
]
SVD_CONFIGS = [
    ("SVD vanilla (cds=1, eig=0)", "vanilla", dict(cds=1.0, eig=0.0, w_plus_c=False)),
    ("SVD cds=0.75 eig=0", "variant", dict(cds=0.75, eig=0.0, w_plus_c=False)),
    ("SVD cds=0.75 eig=0.5", "recommended", dict(cds=0.75, eig=0.5, w_plus_c=False)),
    ("SVD cds=0.75 eig=0.5 w+c", "variant", dict(cds=0.75, eig=0.5, w_plus_c=True)),
    ("SVD cds=0.75 eig=1", "variant", dict(cds=0.75, eig=1.0, w_plus_c=False)),
]
SGNS_CONFIGS = [
    ("SGNS vanilla (neg=1, cds=1)", "vanilla", dict(negative=1, ns_exponent=1.0, w_plus_c=False)),
    ("SGNS neg=5 cds=0.75", "variant", dict(negative=5, ns_exponent=0.75, w_plus_c=False)),
    ("SGNS neg=15 cds=0.75", "recommended", dict(negative=15, ns_exponent=0.75, w_plus_c=False)),
    ("SGNS neg=15 cds=0.75 w+c", "variant", dict(negative=15, ns_exponent=0.75, w_plus_c=True)),
]


def log(msg, t0):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


def cached(path, build, save, load):
    if path.exists():
        return load(path)
    value = build()
    # Write to a temporary name and rename, so an interrupted run never leaves a partial cache.
    tmp = path.with_name(path.stem + ".partial.npz")
    save(tmp, value)
    tmp.replace(path)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="defaults to <data-dir>/cache; holds large matrices and vectors")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--windows", type=int, nargs="+", default=[2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--restrict-analogy", type=int, default=30_000,
                        help="vocabulary size (most frequent words) for the primary analogy metric")
    args = parser.parse_args()

    t0 = time.time()
    cache = args.cache_dir or args.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"min{args.min_count}"

    vocab, ids, n_raw = load_corpus(args.data_dir / "text8" / "text8", min_count=args.min_count)
    log(f"corpus {n_raw} tokens, kept {len(ids)}; vocabulary {len(vocab)} words", t0)
    sim_sets = {k: load_similarity(args.data_dir / "eval-similarity" / f) for k, f in SIMILARITY_SETS.items()}
    sim_sets_totals = {k: len(v) for k, v in sim_sets.items()}
    questions = load_analogies(args.data_dir / "text8" / "questions-words.txt")

    rows = []
    csv_path = args.out / "results.csv"

    def evaluate(name, method, role, win, params, matrix, build_seconds):
        start = time.time()
        space = Space(vocab, matrix)
        params_text = " ".join(f"{k}={v}" for k, v in params.items())
        for key, pairs in sim_sets.items():
            rho, n = evaluate_similarity(space, pairs)
            rows.append(dict(config=name, method=method, role=role, win=win, params=params_text,
                             metric=key, value=rho, n_evaluated=n, n_total=sim_sets_totals[key],
                             build_seconds=round(build_seconds, 1)))
        # Primary analogy metric: questions and candidates limited to the most frequent words.
        # Secondary: the whole vocabulary, where poorly estimated rare words dominate.
        for suffix, restrict in ((f"_{args.restrict_analogy // 1000}k", args.restrict_analogy), ("", None)):
            acc_add, acc_mul, n = evaluate_analogy(space, questions, restrict=restrict)
            for key, acc in (("google_add" + suffix, acc_add), ("google_mul" + suffix, acc_mul)):
                rows.append(dict(config=name, method=method, role=role, win=win, params=params_text,
                                 metric=key, value=acc, n_evaluated=n, n_total=len(questions),
                                 build_seconds=round(build_seconds, 1)))
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
        summary = {r["metric"]: round(r["value"], 3) for r in rows if r["config"] == name and r["win"] == win}
        log(f"win={win} {name}: {summary} (eval {time.time() - start:.0f} s)", t0)

    for win in args.windows:
        start = time.time()
        cooc = cached(
            cache / f"cooc_{tag}_win{win}.npz",
            lambda: cooccurrence(ids, len(vocab), window=win, weighting="uniform"),
            sp.save_npz, sp.load_npz,
        )
        log(f"win={win} co-occurrence matrix nnz={cooc.nnz} ({time.time() - start:.0f} s)", t0)

        for cds in (1.0, 0.75):
            for name, role, params in PPMI_CONFIGS:
                if params["cds"] != cds:
                    continue
                start = time.time()
                matrix = ppmi(cooc, cds=cds, neg=params["neg"])
                evaluate(name, "PPMI", role, win, params, matrix, time.time() - start)
                del matrix
                gc.collect()

            start = time.time()
            factors = cached(
                cache / f"svd_{tag}_win{win}_cds{cds}_dim{args.dim}.npz",
                lambda: svd_factors(ppmi(cooc, cds=cds), args.dim, seed=args.seed),
                lambda p, f: np.savez(p, u=f[0], s=f[1], vt=f[2]),
                lambda p: tuple(np.load(p)[k] for k in ("u", "s", "vt")),
            )
            svd_seconds = time.time() - start
            log(f"win={win} SVD cds={cds} dim={args.dim} ({svd_seconds:.0f} s)", t0)
            for name, role, params in SVD_CONFIGS:
                if params["cds"] == cds:
                    vectors = svd_vectors(*factors, eig=params["eig"], w_plus_c=params["w_plus_c"])
                    evaluate(name, "SVD", role, win, params, vectors, svd_seconds)
            del factors
            gc.collect()
        del cooc
        gc.collect()

        trained = {}
        for name, role, params in SGNS_CONFIGS:
            key = (params["negative"], params["ns_exponent"])
            if key not in trained:
                start = time.time()
                path = cache / (f"sgns_{tag}_win{win}_neg{key[0]}_ns{key[1]}_dim{args.dim}"
                                f"_ep{args.epochs}_seed{args.seed}.npz")
                trained = {key: cached(
                    path,
                    lambda: train_sgns(vocab, ids, negative=key[0], ns_exponent=key[1], window=win,
                                       dim=args.dim, epochs=args.epochs, workers=args.workers,
                                       seed=args.seed),
                    lambda p, v: np.savez(p, w=v[0], c=v[1]),
                    lambda p: tuple(np.load(p)[k] for k in ("w", "c")),
                )}
                sgns_seconds = time.time() - start
                log(f"win={win} SGNS neg={key[0]} ns_exponent={key[1]} ({sgns_seconds:.0f} s)", t0)
            word, context = trained[key]
            vectors = word + context if params["w_plus_c"] else word
            evaluate(name, "SGNS", role, win, params, vectors, sgns_seconds)

    results = pd.DataFrame(rows)
    results.to_csv(csv_path, index=False, float_format="%.4f")
    poster_chart(results, args.out / "poster_chart.png", window=args.windows[0])
    log(f"done: {csv_path} and {args.out / 'poster_chart.png'}", t0)


if __name__ == "__main__":
    main()
