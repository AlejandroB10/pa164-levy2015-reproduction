"""Combine scorers: weighted sum of per-source standardized scores.

For one source, each component scores all candidates; each score vector is turned into
z-scores over the candidates (so cosine, counts and log-frequencies live on one scale),
then summed with weights. -inf (never seen) becomes the component's lowest finite value.

Typical uses:
* embedding + popularity prior:  CombinedScorer([emb, LogFreq(train)], [1, lam])
* pair-freq back-off:            CombinedScorer([LogPairCount(train), emb], [big, 1])
  -> seen (source, target) pairs come first, ordered by count; the embedding orders the rest.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .data import SubPair


def _z(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    s[~finite] = s[finite].min()
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)


class LogFreq:
    """log(1 + how often the candidate is a substitution target in train)."""

    name = "logfreq"
    vocab = None

    def __init__(self, train: list[SubPair]):
        self.count = Counter(p.target for p in train)

    def scores(self, source, candidates):
        return np.array([math.log1p(self.count.get(c, 0)) for c in candidates])


class LogPairCount:
    """log(1 + count of (source, candidate) in train)."""

    name = "logpair"
    vocab = None

    def __init__(self, train: list[SubPair]):
        self.count = Counter((p.source, p.target) for p in train)

    def scores(self, source, candidates):
        return np.array([math.log1p(self.count.get((source, c), 0)) for c in candidates])


class CombinedScorer:
    def __init__(self, components, weights, name=None, raw=None):
        """``raw``: indices of components added without z-scoring (e.g. a back-off whose
        absolute scale must dominate)."""
        if len(components) != len(weights):
            raise ValueError("one weight per component")
        self.components, self.weights = list(components), list(weights)
        self.raw = set(raw or ())
        self.name = name or "+".join(f"{w:g}*{c.name}" for c, w in zip(components, weights))
        vocabs = [c.vocab for c in components if c.vocab is not None]
        self.vocab = set.intersection(*map(set, vocabs)) if vocabs else None
        self._cache = {}

    def component_scores(self, source, candidates):
        key = (source, len(candidates), candidates[0] if candidates else None)
        if key not in self._cache:
            out = []
            for i, c in enumerate(self.components):
                s = c.scores(source, candidates)
                out.append(np.asarray(s, dtype=np.float64) if i in self.raw else _z(s))
            self._cache = {key: out}  # keep only the latest source
        return self._cache[key]

    def scores(self, source, candidates):
        parts = self.component_scores(source, candidates)
        return sum(w * p for w, p in zip(self.weights, parts))
