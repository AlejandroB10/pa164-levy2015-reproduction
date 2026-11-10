"""Recipe1MSubs (GISMo) ingredient vocabulary and substitution pairs.

The GISMo files are pickles. Unpickling can run arbitrary code, so they are read with an
unpickler that only rebuilds plain containers and the one class the vocabulary needs.
"""
import pickle
from dataclasses import dataclass

ALLOWED = {("inv_cooking.datasets.vocabulary", "Vocabulary")}


class _Vocabulary:
    """Stand-in for GISMo's Vocabulary: only its attributes are needed."""


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return _Vocabulary
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def safe_load(file):
    return _SafeUnpickler(file).load()


@dataclass
class IngredientVocabulary:
    canonical: list    # one canonical name per cluster, in cluster-id order
    cluster_id: dict   # canonical name -> GISMo cluster id
    surface: dict      # every surface form -> canonical name


def load_vocabulary(path):
    """Clusters of `vocab_ingrs.pkl`. The canonical name is the cluster's first surface form."""
    with open(path, "rb") as f:
        raw = safe_load(f)
    canonical, cluster_id, surface = [], {}, {}
    for idx in sorted(raw.idx2word):
        forms = raw.idx2word[idx]
        if idx == 0:  # <end> token
            continue
        forms = forms if isinstance(forms, list) else [forms]
        head = forms[0]
        canonical.append(head)
        cluster_id[head] = idx
    for form, idx in raw.word2idx.items():
        if idx != 0:
            forms = raw.idx2word[idx]
            surface[form] = forms[0] if isinstance(forms, list) else forms
    return IngredientVocabulary(canonical, cluster_id, surface)


def load_substitutions(path, vocab):
    """(original, substitute) pairs as canonical names, one per annotated recipe comment."""
    with open(path, "rb") as f:
        rows = safe_load(f)
    return [(vocab.surface[a], vocab.surface[b]) for a, b in (r["subs"] for r in rows)]
