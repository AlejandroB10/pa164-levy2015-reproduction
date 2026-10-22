"""Word-context co-occurrence counting with a fixed symmetric window."""
import numpy as np
import scipy.sparse as sp


def cooccurrence(ids, vocab_size, window, weighting="uniform"):
    """Symmetric word-context count matrix over a token-id stream.

    weighting="uniform": every context within `window` counts 1 (the paper's dyn=none).
    weighting="harmonic": a context at distance d counts 1/d (used only by the sanity check,
    to mirror the earlier prototype).
    Words and contexts share one vocabulary, so the matrix is square.
    """
    if weighting not in ("uniform", "harmonic"):
        raise ValueError(f"unknown weighting {weighting!r}")
    shape = (vocab_size, vocab_size)
    total = sp.csr_matrix(shape, dtype=np.float64)
    for d in range(1, window + 1):
        w = 1.0 if weighting == "uniform" else 1.0 / d
        data = np.full(len(ids) - d, w, dtype=np.float64)
        total = total + sp.coo_matrix((data, (ids[:-d], ids[d:])), shape=shape).tocsr()
    # Left contexts of a word are the right contexts read in reverse, hence the transpose.
    return (total + total.T).tocsr()
