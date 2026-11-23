"""PROVISIONAL classic ingredient embeddings (PPMI, SVD, SGNS) from recipe ingredient lists.

Stand-in until the team's Open Recipes models arrive: same methods and knobs as Levy et al.
(2015), trained here on the ingredient lists of the Recipe1MSubs *train* recipes, which are
already local. Context = the whole recipe (a bag of ingredients), so two ingredients are
similar when they appear with the same other ingredients (2nd-order similarity, which is what
substitutes share), unlike the PMI baseline, which scores direct co-occurrence (complements).

    python classic_recipes.py --data-dir ../data/recipe1msubs --grid --out results/val_classic_grid.csv
    python classic_recipes.py --data-dir ../data/recipe1msubs --method svd --dim 300 --eig 0.5 \
        --cds 0.75 --out-vectors vectors/svd_recipes.npz
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import svds  # noqa: E402

from subeval.data import load_split, load_vocab, recipes_from_pairs  # noqa: E402
from subeval.evaluate import rank_pairs  # noqa: E402
from subeval.metrics import summarize  # noqa: E402
from subeval.scorers import EmbeddingScorer  # noqa: E402


def build_vocab(recipes, min_count=1):
    cnt = Counter(i for r in recipes for i in set(r))
    words = sorted(w for w, c in cnt.items() if c >= min_count)
    return words, {w: i for i, w in enumerate(words)}


def cooccurrence(recipes, index):
    """Symmetric ingredient x ingredient counts, one count per pair of distinct ingredients per recipe."""
    rows, cols = [], []
    for r in recipes:
        ids = sorted({index[i] for i in r if i in index})
        for a, b in itertools.permutations(ids, 2):
            rows.append(a)
            cols.append(b)
    n = len(index)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))


def ppmi(cooc, cds=1.0, neg=1):
    """Shifted PPMI with context distribution smoothing: max(PMI_alpha - log k, 0)."""
    cooc = cooc.tocoo()
    total = cooc.sum()
    w_sum = np.asarray(cooc.sum(1)).ravel()
    c_cnt = np.asarray(cooc.sum(0)).ravel() ** cds
    p_c = c_cnt / c_cnt.sum()
    p_w = w_sum / total
    p_wc = cooc.data / total
    pmi = np.log(p_wc / (p_w[cooc.row] * p_c[cooc.col])) - np.log(neg)
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (cooc.row[keep], cooc.col[keep])), shape=cooc.shape)


def svd_vectors(m, dim, eig=0.5, wc=False, seed=0):
    k = min(dim, min(m.shape) - 1)
    u, s, vt = svds(m, k=k, random_state=seed)
    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order]
    w = u * s**eig
    return w + vt.T * s**eig if wc else w


def sgns_vectors(recipes, words, dim=100, neg=5, cds=0.75, epochs=20, wc=False, seed=0, workers=8):
    from gensim.models import Word2Vec

    rng = random.Random(seed)
    sents = []
    for r in recipes:
        s = list(dict.fromkeys(r))  # unique, order kept
        rng.shuffle(s)  # order in an ingredient list carries little meaning
        sents.append(s)
    window = max(len(s) for s in sents)  # whole recipe = context
    model = Word2Vec(sents, vector_size=dim, window=window, min_count=1, sg=1, hs=0, negative=neg,
                     ns_exponent=cds, sample=0, shrink_windows=False, epochs=epochs, seed=seed,
                     workers=workers)
    idx = [model.wv.key_to_index[w] for w in words]
    w = model.wv.vectors[idx]
    return w + model.syn1neg[idx] if wc else w


def evaluate_val(words, vecs, val, candidates):
    sc = EmbeddingScorer(words, vecs)
    cands = [c for c in candidates if c in sc.vocab]
    return summarize(rank_pairs(sc, val, cands)), len(cands)


def grid(args):
    vocab = load_vocab(args.data_dir)
    train = load_split(args.data_dir, "train", vocab)
    val = [p for p in load_split(args.data_dir, "val", vocab) if p.source != p.target]
    recipes = recipes_from_pairs(train)
    words, index = build_vocab(recipes)
    cooc = cooccurrence(recipes, index)
    print(f"{len(recipes)} train recipes, {len(words)} ingredients, {cooc.nnz} non-zero pairs", flush=True)

    fields = ["method", "cds", "neg", "dim", "eig", "wc", "epochs", "mrr", "hit@1", "hit@3", "hit@10", "n_cands"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        def log(row, vecs):
            t0 = time.time()
            m, nc = evaluate_val(words, vecs, val, words)
            row.update({k: round(m[k], 2) for k in ("mrr", "hit@1", "hit@3", "hit@10")}, n_cands=nc)
            w.writerow({k: row.get(k, "") for k in fields})
            f.flush()
            print(f"{row}  ({time.time() - t0:.0f} s eval)", flush=True)

        for cds in (1.0, 0.75):
            for neg in (1, 5):
                m = ppmi(cooc, cds, neg)
                log({"method": "ppmi", "cds": cds, "neg": neg}, m.toarray().astype(np.float32))
                for dim in (100, 300):
                    for eig in (0.0, 0.5, 1.0):
                        for wc in (False, True):
                            log({"method": "svd", "cds": cds, "neg": neg, "dim": dim, "eig": eig, "wc": wc},
                                svd_vectors(m, dim, eig, wc))
        for dim in (100, 300):
            for neg in (5, 15):
                for cds in (1.0, 0.75):
                    for epochs in (10, 30):
                        for wc in (False, True):
                            log({"method": "sgns", "cds": cds, "neg": neg, "dim": dim, "epochs": epochs, "wc": wc},
                                sgns_vectors(recipes, words, dim, neg, cds, epochs, wc))


def export(args):
    vocab = load_vocab(args.data_dir)
    recipes = recipes_from_pairs(load_split(args.data_dir, "train", vocab))
    words, index = build_vocab(recipes)
    if args.method == "sgns":
        vecs = sgns_vectors(recipes, words, args.dim, args.neg, args.cds, args.epochs, args.wc)
    else:
        m = ppmi(cooccurrence(recipes, index), args.cds, args.neg)
        vecs = m.toarray().astype(np.float32) if args.method == "ppmi" else svd_vectors(m, args.dim, args.eig, args.wc)
    Path(args.out_vectors).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_vectors, words=np.array(words), vectors=np.asarray(vecs, dtype=np.float32))
    print(f"{len(words)} x {vecs.shape[1]} -> {args.out_vectors}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--grid", action="store_true", help="tune every method on val, write --out")
    ap.add_argument("--out", help="grid CSV")
    ap.add_argument("--method", choices=["ppmi", "svd", "sgns"])
    ap.add_argument("--cds", type=float, default=0.75)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--eig", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--wc", action="store_true", help="w+c: add context vectors")
    ap.add_argument("--out-vectors")
    args = ap.parse_args()
    if args.grid:
        if not args.out:
            ap.error("--grid needs --out")
        grid(args)
    else:
        if not (args.method and args.out_vectors):
            ap.error("give --method and --out-vectors (or --grid)")
        export(args)


if __name__ == "__main__":
    main()
