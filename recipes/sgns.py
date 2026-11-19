"""SGNS (gensim skip-gram with negative sampling) trained on recipes.

A recipe is a set of ingredients, so any two ingredients in the same recipe form a
(word, context) pair. That is why the window spans the longest recipe and never shrinks: the
order in which someone listed the ingredients means nothing.
"""
from gensim.models import Word2Vec

# gensim's own defaults, spelled out so the report can say exactly what "vanilla" means.
DEFAULTS = dict(vector_size=100, negative=5, ns_exponent=0.75, sample=1e-3, epochs=5,
                alpha=0.025, min_count=5)


def read_recipes(corpus_path):
    """Read the corpus: one recipe per line, canonical ingredients separated by spaces."""
    with open(corpus_path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def train_sgns(corpus_path, seed=1, workers=1, **params):
    """Train SGNS. Anything in `params` overrides DEFAULTS (vector_size, negative, ...).

    For a reproducible run, use workers=1 and set PYTHONHASHSEED before Python starts.
    """
    recipes = read_recipes(corpus_path)
    config = {**DEFAULTS, **params}
    return Word2Vec(recipes, sg=1, hs=0, window=max(map(len, recipes)), shrink_windows=False,
                    seed=seed, workers=workers, **config)


def vectors(model, w_plus_c=False):
    """Return (names, matrix) with the word vectors W, or W + C (the paper's w+c)."""
    matrix = model.wv.vectors
    if w_plus_c:
        matrix = matrix + model.syn1neg  # row i of syn1neg is the context vector of word i
    return list(model.wv.index_to_key), matrix
