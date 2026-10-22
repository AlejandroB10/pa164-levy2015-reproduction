"""Sanity check: rebuild the earlier prototype setting with this package's code.

Prototype setting: top-30k vocabulary, window 5 with 1/d weights, PPMI with cds=0.75,
300-dim SVD, eig=0.5, Google analogies (3CosAdd) with candidates restricted to the 30k
words. The prototype, evaluated with gensim, reported 44.0 %.

Usage:
    python sanity_check.py --data-dir PATH/TO/ML-NLP-data
"""
import argparse
import sys
import time
from pathlib import Path

from repro.corpus import load_corpus
from repro.counts import cooccurrence
from repro.evaluation import Space, evaluate_analogy, load_analogies
from repro.ppmi_svd import ppmi, svd_factors, svd_vectors

VOCAB_SIZE = 30_000
EXPECTED_ADD = 0.440


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.time()
    vocab, ids, _ = load_corpus(args.data_dir / "text8" / "text8", top_n=VOCAB_SIZE)
    matrix = ppmi(cooccurrence(ids, len(vocab), window=5, weighting="harmonic"), cds=0.75)
    u, s, vt = svd_factors(matrix, dim=300)
    space = Space(vocab, svd_vectors(u, s, vt, eig=0.5))
    questions = load_analogies(args.data_dir / "text8" / "questions-words.txt")
    acc_add, acc_mul, n = evaluate_analogy(space, questions, restrict=VOCAB_SIZE)
    print(
        f"sanity PPMI+SVD 30k: 3CosAdd {acc_add:.3f}  3CosMul {acc_mul:.3f}  "
        f"questions {n}/{len(questions)}  ({time.time() - t0:.0f} s)"
    )
    if round(acc_add, 3) != EXPECTED_ADD:
        sys.exit(f"FAIL: expected 3CosAdd {EXPECTED_ADD}, got {acc_add:.3f}")
    print("OK: pipeline reproduces the prototype number.")


if __name__ == "__main__":
    main()
