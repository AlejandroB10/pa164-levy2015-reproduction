"""Embedding file format shared by the whole team.

It is plain word2vec text: a "<rows> <dims>" header, then one "<ingredient> <v1> ... <vd>"
line per ingredient. gensim can read it with KeyedVectors.load_word2vec_format, which lets us
score PPMI/SVD, SGNS and the transformers with the same evaluation code.
"""
import numpy as np


def save_embeddings(path, names, vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(names) != len(vectors):
        raise ValueError(f"{len(names)} names but vectors of shape {vectors.shape}")
    for name in names:
        if not name or any(ch.isspace() for ch in name):
            raise ValueError(f"ingredient name must be one non-empty token: {name!r}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(names)} {vectors.shape[1]}\n")
        for name, row in zip(names, vectors):
            f.write(name + " " + " ".join(f"{x:.7g}" for x in row) + "\n")


def load_embeddings(path):
    """Return (names, float32 matrix), in the order of the file."""
    with open(path, encoding="utf-8") as f:
        rows, dims = map(int, f.readline().split())
        names, vectors = [], np.empty((rows, dims), dtype=np.float32)
        for i, line in enumerate(f):
            name, *values = line.rstrip("\n").split(" ")
            if i >= rows or len(values) != dims:
                raise ValueError(f"{path}: header says {rows} rows of {dims} values, "
                                 f"line {i + 2} ({name!r}) does not fit")
            names.append(name)
            vectors[i] = np.array(values, dtype=np.float32)
    if len(names) != rows:
        raise ValueError(f"{path}: header says {rows} rows but the file has {len(names)}")
    return names, vectors
