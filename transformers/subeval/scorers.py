"""Things that score candidate substitutes for a source ingredient.

Every scorer has ``vocab`` (the ingredient names it can score, normalized) and
``scores(source, candidates) -> np.ndarray`` (higher = better substitute). Embeddings use
cosine similarity; the baselines are context-free counts, like GISMo's LT/Freq baselines.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from .data import SubPair, normalize


# ---------------------------------------------------------------- embeddings


def load_vectors(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Shared embedding format: ingredient -> vector.

    * ``.txt`` / ``.vec``: word2vec text format, one ``name v1 v2 ...`` per line, optional
      ``n_words dim`` header. Names may not contain spaces: use '_' (``olive_oil``).
    * ``.npz``: arrays ``words`` (str) and ``vectors`` (n x d).
    Names are normalized; if two names collide the first one is kept.
    """
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        words, mat = [str(w) for w in z["words"]], np.asarray(z["vectors"], dtype=np.float32)
    else:
        words, rows = [], []
        with open(path, encoding="utf-8") as f:
            first = f.readline().split()
            if not (len(first) == 2 and all(t.isdigit() for t in first)):
                words.append(first[0])
                rows.append([float(x) for x in first[1:]])
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                words.append(parts[0])
                rows.append([float(x) for x in parts[1:]])
        mat = np.asarray(rows, dtype=np.float32)
    if len(words) != mat.shape[0]:
        raise ValueError(f"{path}: {len(words)} names but {mat.shape[0]} vectors")
    keep, seen = [], set()
    for i, w in enumerate(words):
        k = normalize(w)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return [normalize(words[i]) for i in keep], mat[keep]


class EmbeddingScorer:
    """Cosine similarity between L2-normalized ingredient vectors."""

    def __init__(self, words: list[str], vectors: np.ndarray, name: str = "embedding"):
        self.name = name
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        ok = norms[:, 0] > 0  # an all-zero vector has no direction: treat as not covered
        self.index = {w: i for i, w in enumerate(np.asarray(words)[ok])}
        self.mat = (vectors[ok] / norms[ok]).astype(np.float32)
        self.vocab = set(self.index)

    @classmethod
    def from_file(cls, path, name=None):
        words, mat = load_vectors(path)
        return cls(words, mat, name or Path(path).stem)

    def scores(self, source: str, candidates: list[str]) -> np.ndarray:
        idx = [self.index[c] for c in candidates]
        return self.mat[idx] @ self.mat[self.index[source]]


# ---------------------------------------------------------------- baselines


class FrequencyScorer:
    """Popularity: how often each ingredient is the *target* of a substitution in train."""

    name = "freq"

    def __init__(self, train: list[SubPair]):
        self.count = Counter(p.target for p in train)
        self.vocab = None  # scores any name (unseen = 0)

    def scores(self, source, candidates):
        return np.array([self.count.get(c, 0) for c in candidates], dtype=np.float64)


class PairFrequencyScorer:
    """GISMo's lookup-table baseline: count of (source -> candidate) in train,
    ties broken by target popularity."""

    name = "pair-freq"

    def __init__(self, train: list[SubPair]):
        self.pairs = Counter((p.source, p.target) for p in train)
        self.pop = Counter(p.target for p in train)
        self.vocab = None

    def scores(self, source, candidates):
        top = max(self.pop.values(), default=0) + 1
        return np.array(
            [self.pairs.get((source, c), 0) + self.pop.get(c, 0) / top for c in candidates],
            dtype=np.float64,
        )


class GismoTableScorer:
    """GISMo's own baselines, exactly as in gismo/models.py (to reproduce their Table 1).

    * ``gismo-freq``: ``torch.sum(train_table, 1)`` = how often a candidate is a *source* in
      train (the paper's text says "target"; the code counts sources).
    * ``gismo-lt``: 1 if (source, candidate) appears in train, else 0.
    * ``gismo-ltfreq``: raw count of (source, candidate) in train, no tie-breaking.
    """

    def __init__(self, train: list[SubPair], kind: str):
        if kind not in ("gismo-freq", "gismo-lt", "gismo-ltfreq"):
            raise ValueError(kind)
        self.name, self.kind = kind, kind
        self.src = Counter(p.source for p in train)
        self.pairs = Counter((p.source, p.target) for p in train)
        self.vocab = None

    def scores(self, source, candidates):
        if self.kind == "gismo-freq":
            vals = [self.src.get(c, 0) for c in candidates]
        elif self.kind == "gismo-lt":
            vals = [1 if self.pairs.get((source, c), 0) else 0 for c in candidates]
        else:
            vals = [self.pairs.get((source, c), 0) for c in candidates]
        return np.array(vals, dtype=np.float64)


class PMIScorer:
    """PMI of two ingredients appearing in the same recipe: PMI(a,b) = log p(a,b) / p(a)p(b).

    Co-occurrence points to complements rather than substitutes, which makes this an
    informative weak baseline. Pairs never seen together get -inf (ties -> pessimistic rank).
    """

    name = "pmi"

    def __init__(self, recipes, min_count: int = 1):
        self.n = 0
        self.single = Counter()
        self.joint = Counter()
        for rec in recipes:
            items = sorted(set(normalize(x) for x in rec))
            if not items:
                continue
            self.n += 1
            self.single.update(items)
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    self.joint[(a, b)] += 1
        self.min_count = min_count
        # A baseline must not shrink the shared candidate set: unseen names stay candidates
        # and simply score -inf.
        self.vocab = None

    def pmi(self, a, b):
        key = (a, b) if a < b else (b, a)
        j = self.joint.get(key, 0)
        if j == 0 or self.single.get(a, 0) < self.min_count or self.single.get(b, 0) < self.min_count:
            return -math.inf
        return math.log(j * self.n / (self.single[a] * self.single[b]))

    def scores(self, source, candidates):
        return np.array([self.pmi(source, c) for c in candidates], dtype=np.float64)
