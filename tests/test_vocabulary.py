import io
import os
import pickle
from pathlib import Path

import pytest

from recipes.vocabulary import load_substitutions, load_vocabulary, safe_load

DATA = Path(os.environ.get("RECIPES_DATA", "data")) / "recipe1msubs"
needs_data = pytest.mark.skipif(not (DATA / "vocab_ingrs.pkl").exists(), reason="run ./download_recipes.sh")


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_load_refuses_unknown_classes():
    with pytest.raises(pickle.UnpicklingError):
        safe_load(io.BytesIO(pickle.dumps(Evil())))


def test_safe_load_accepts_plain_containers():
    assert safe_load(io.BytesIO(pickle.dumps({"a": [("b", 1)]}))) == {"a": [("b", 1)]}


@needs_data
def test_vocabulary_maps_every_surface_form_to_its_cluster_head():
    vocab = load_vocabulary(DATA / "vocab_ingrs.pkl")
    assert len(vocab.canonical) == 6633  # 6,634 clusters minus <end>
    assert vocab.surface["garlic_cloves"] == "garlic_clove"
    assert vocab.surface["cloves"] == "clove"
    assert "<end>" not in vocab.surface
    assert vocab.cluster_id["garlic"] == 2602


@needs_data
def test_every_substitution_word_is_in_the_vocabulary():
    vocab = load_vocabulary(DATA / "vocab_ingrs.pkl")
    for split in ("val", "test"):
        pairs = load_substitutions(DATA / f"{split}_comments_subs.pkl", vocab)
        assert pairs and all(a in vocab.canonical and b in vocab.canonical for a, b in pairs)
