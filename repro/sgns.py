"""Skip-gram with negative sampling (gensim), aligned to the shared vocabulary."""
import numpy as np
from gensim.models import Word2Vec

SENTENCE_LENGTH = 10_000  # gensim's per-sentence limit; the stream is cut into chunks of this size


def train_sgns(vocab, ids, *, negative, ns_exponent, window, dim, epochs, workers, seed):
    """Train SGNS on the shared token stream.

    Returns (word_vectors, context_vectors), both with rows in `vocab` order.
    shrink_windows=False gives a fixed (uniform) window, i.e. the paper's dyn=none.
    sample=0 disables subsampling (sub=none). ns_exponent is the cds alpha of SGNS.
    """
    sentences = [
        [vocab[i] for i in ids[start : start + SENTENCE_LENGTH]]
        for start in range(0, len(ids), SENTENCE_LENGTH)
    ]
    model = Word2Vec(
        sentences,
        sg=1,
        hs=0,
        negative=negative,
        ns_exponent=ns_exponent,
        window=window,
        shrink_windows=False,
        vector_size=dim,
        sample=0,
        min_count=1,  # the stream is already restricted to the shared vocabulary
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    if set(model.wv.index_to_key) != set(vocab):
        raise RuntimeError("gensim vocabulary differs from the shared vocabulary")
    order = np.array([model.wv.key_to_index[w] for w in vocab])
    return model.wv.vectors[order].copy(), model.syn1neg[order].copy()
