"""Ranking metrics as in GISMo (recommender_gcn.py): MRR and Hit@k, reported in %.

Rank of the true target among the candidates, 1 = best. Ties count against the target
(GISMo counts every candidate with score >= the target's), so a constant scorer gets the
worst rank, not a lucky one.
"""

from __future__ import annotations

import numpy as np

KS = (1, 3, 10)


def rank_of(scores: np.ndarray, target: int, exclude=()) -> int:
    """1-based pessimistic rank of ``target``; candidates in ``exclude`` are left out."""
    s = np.asarray(scores, dtype=np.float64).copy()
    t = s[target]
    if np.isnan(t):
        raise ValueError("target score is NaN")
    excl = [i for i in exclude if i != target]
    if excl:
        s[excl] = -np.inf
    s[target] = -np.inf  # do not count the target against itself
    return 1 + int(np.sum(s >= t))


def summarize(ranks) -> dict:
    """MRR and Hit@k in %. A rank of 0 or None means "not covered" and counts as a miss."""
    r = np.array([x if x else np.inf for x in ranks], dtype=np.float64)
    n = len(r)
    if n == 0:
        return {"n": 0, "mrr": float("nan"), **{f"hit@{k}": float("nan") for k in KS}}
    out = {"n": n, "mrr": float(np.mean(1.0 / r) * 100)}
    for k in KS:
        out[f"hit@{k}"] = float(np.mean(r <= k) * 100)
    return out
