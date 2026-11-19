"""Substitution ranking on Recipe1MSubs pairs, used for every model.

For each (target, substitute) pair we rank every ingredient that has a vector, except the
target, by cosine similarity to the target, and look at where the true substitute lands. That
rank gives MRR and Hit@k. If the model has no vector for the target or the substitute, the pair
scores 0, so a model cannot look better just by leaving out rare ingredients. An all-zero or
non-finite row counts as no vector, and ties count against the model.
"""
from collections import defaultdict

import numpy as np

KS = (1, 3, 10)


def substitute_ranks(names, matrix, pairs):
    """Rank of each pair's substitute (1 is best), or None if the pair is not covered."""
    norms = np.linalg.norm(matrix, axis=1)
    # An all-zero or non-finite row carries no information, so it counts as no vector at all.
    usable = np.isfinite(matrix).all(axis=1) & (norms > 0)
    matrix, names = matrix[usable], [n for n, ok in zip(names, usable) if ok]
    index = {name: i for i, name in enumerate(names)}
    unit = matrix / norms[usable][:, None]
    ranks = [None] * len(pairs)
    by_target = defaultdict(list)
    for k, (target, sub) in enumerate(pairs):
        if target in index and sub in index and target != sub:
            by_target[index[target]].append((k, index[sub]))
    for t, items in by_target.items():
        sims = unit @ unit[t]
        sims[t] = -np.inf  # a target can't be its own substitute
        for k, s in items:
            # ties count against the model: the substitute ranks after every equal candidate
            ranks[k] = int(np.sum(sims >= sims[s]))
    return ranks


def evaluate_substitutions(names, matrix, pairs):
    """MRR and Hit@k over all pairs (uncovered ones count as 0), covered-only MRR, coverage."""
    ranks = substitute_ranks(names, np.asarray(matrix, dtype=np.float64), pairs)
    covered = [r for r in ranks if r is not None]
    n = len(ranks)
    metrics = {"n": n, "coverage": len(covered) / n,
               "mrr": sum(1 / r for r in covered) / n,
               "mrr_covered": sum(1 / r for r in covered) / len(covered) if covered else 0.0}
    for k in KS:
        metrics[f"hit@{k}"] = sum(r <= k for r in covered) / n
    return metrics
