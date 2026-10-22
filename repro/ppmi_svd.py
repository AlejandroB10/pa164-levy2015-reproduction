"""PPMI with context-distribution smoothing and shifting, and SVD factorisation."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def ppmi(cooc, cds=1.0, neg=1):
    """Shifted PPMI: max(PMI_alpha(w, c) - log(neg), 0) as a float32 CSR matrix.

    PMI_alpha(w, c) = log( #(w,c) / #(w) / P_alpha(c) ), where
    P_alpha(c) = #(c)^alpha / sum_c' #(c')^alpha  (cds = alpha; alpha = 1 is plain PMI).
    """
    sum_w = np.asarray(cooc.sum(axis=1)).ravel()
    sum_c = np.asarray(cooc.sum(axis=0)).ravel() ** cds
    p_c = sum_c / sum_c.sum()
    coo = cooc.tocoo()
    pmi = np.log(coo.data) - np.log(sum_w[coo.row]) - np.log(p_c[coo.col]) - np.log(neg)
    keep = pmi > 0
    return sp.csr_matrix(
        (pmi[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=cooc.shape
    )


def svd_factors(matrix, dim, seed=1):
    """Truncated SVD (ARPACK). Returns U, s, Vt with singular values in descending order."""
    rng = np.random.default_rng(seed)
    v0 = rng.random(min(matrix.shape))
    u, s, vt = svds(matrix, k=dim, v0=v0)
    order = np.argsort(-s)
    return u[:, order].astype(np.float32), s[order].astype(np.float32), vt[order].astype(np.float32)


def svd_vectors(u, s, vt, eig, w_plus_c=False):
    """Word vectors U * S^eig; with w+c also add the context vectors V * S^eig."""
    weights = s**eig
    vectors = u * weights
    if w_plus_c:
        vectors = vectors + vt.T * weights
    return vectors
