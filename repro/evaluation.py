"""Word-similarity (Spearman rho) and analogy (3CosAdd / 3CosMul) evaluation."""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

SIMILARITY_SETS = {
    "ws_sim": "ws353_similarity.txt",
    "ws_rel": "ws353_relatedness.txt",
    "men": "bruni_men.txt",
    "mturk": "radinsky_mturk.txt",
    "rare_words": "luong_rare.txt",
    "simlex": "simlex999_gensim.txt",
}
MUL_EPSILON = 1e-3  # paper (Sec. 4.3); the hyperwords code uses 0.01
ANALOGY_BATCH = 512


def load_similarity(path):
    """Word pairs with a gold score. Skips '#' comment lines; lowercases words."""
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        x, y, score = line.split()[:3]
        pairs.append((x.lower(), y.lower(), float(score)))
    return pairs


def load_analogies(path):
    """Google analogy questions as (a, a*, b, b*) tuples, lowercased, section headers dropped."""
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(":"):
            continue
        questions.append(tuple(line.lower().split()))
    return questions


class Space:
    """Row-normalised word vectors (dense ndarray or sparse CSR) over a fixed vocabulary."""

    def __init__(self, vocab, matrix):
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.sparse = sp.issparse(matrix)
        if self.sparse:
            m = sp.csr_matrix(matrix, dtype=np.float32)
            norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            self.m = sp.diags(1.0 / norms).dot(m).tocsr().astype(np.float32)
        else:
            m = np.asarray(matrix, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.m = m / norms

    def cosine_block(self, rows):
        """Cosine between the given rows and every word: dense float32 array (len(rows), V)."""
        if self.sparse:
            return self.m[rows].dot(self.m.T).toarray().astype(np.float32)
        return self.m[rows] @ self.m.T


def evaluate_similarity(space, pairs):
    """Spearman rho over pairs with both words in the vocabulary. Returns (rho, n_used)."""
    used = [(x, y, s) for x, y, s in pairs if x in space.index and y in space.index]
    if len(used) < 2:
        return float("nan"), len(used)
    rows_x = [space.index[x] for x, _, _ in used]
    rows_y = [space.index[y] for _, y, _ in used]
    if space.sparse:
        model = np.asarray(space.m[rows_x].multiply(space.m[rows_y]).sum(axis=1)).ravel()
    else:
        model = np.einsum("ij,ij->i", space.m[rows_x], space.m[rows_y])
    rho = spearmanr(model, [s for _, _, s in used]).statistic
    return float(rho), len(used)


def evaluate_analogy(space, questions, restrict=None):
    """3CosAdd and 3CosMul accuracy on questions whose four words are in the vocabulary.

    With `restrict=N`, both the questions (all four words) and the candidate answers are
    limited to the N most frequent words (the vocabulary is frequency-ordered); otherwise the
    whole vocabulary is used. The three question words are never candidates. For 3CosMul, dense cosines are shifted to [0, 1] via (x+1)/2;
    sparse PPMI cosines are already non-negative and are used as is.
    Returns (acc_add, acc_mul, n_used).
    """
    limit = len(space.vocab) if restrict is None else restrict
    used = [q for q in questions if all(w in space.index and space.index[w] < limit for w in q)]
    if not used:
        return float("nan"), float("nan"), 0
    words = sorted({w for q in used for w in q[:3]})
    local = {w: i for i, w in enumerate(words)}
    sims = space.cosine_block([space.index[w] for w in words])[:, :limit]
    mul_sims = sims if space.sparse else (sims + 1.0) / 2.0

    q_local = np.array([[local[a], local[a_], local[b]] for a, a_, b, _ in used])
    q_vocab = np.array([[space.index[w] for w in q] for q in used])
    correct_add = correct_mul = 0
    for start in range(0, len(used), ANALOGY_BATCH):
        ql = q_local[start : start + ANALOGY_BATCH]
        qv = q_vocab[start : start + ANALOGY_BATCH]
        rows = np.arange(len(ql))[:, None]
        add = sims[ql[:, 2]] - sims[ql[:, 0]] + sims[ql[:, 1]]
        mul = mul_sims[ql[:, 1]] * mul_sims[ql[:, 2]] / (mul_sims[ql[:, 0]] + MUL_EPSILON)
        add[rows, qv[:, :3]] = -np.inf
        mul[rows, qv[:, :3]] = -np.inf
        correct_add += int((add.argmax(axis=1) == qv[:, 3]).sum())
        correct_mul += int((mul.argmax(axis=1) == qv[:, 3]).sum())
    return correct_add / len(used), correct_mul / len(used), len(used)
