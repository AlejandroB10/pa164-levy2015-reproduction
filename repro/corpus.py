"""Corpus loading, vocabulary selection and integer encoding."""
from collections import Counter

import numpy as np


def load_corpus(path, min_count=None, top_n=None):
    """Read a whitespace-tokenised corpus and encode it against one shared vocabulary.

    The vocabulary is ordered by descending frequency (ties broken by first occurrence),
    so the first N rows of any matrix built on it are the N most frequent words.
    Out-of-vocabulary tokens are deleted from the stream before any counting or training,
    so every method sees exactly the same token sequence.

    Returns (vocab, ids, n_raw_tokens).
    """
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()
    counts = Counter(tokens)
    ranked = counts.most_common()
    if min_count is not None:
        ranked = [(w, c) for w, c in ranked if c >= min_count]
    if top_n is not None:
        ranked = ranked[:top_n]
    vocab = [w for w, _ in ranked]
    index = {w: i for i, w in enumerate(vocab)}
    ids = np.fromiter((index[t] for t in tokens if t in index), dtype=np.int32)
    return vocab, ids, len(tokens)
