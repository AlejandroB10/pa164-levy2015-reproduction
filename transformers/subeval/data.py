"""Load the Recipe1MSubs splits (GISMo, Fatemi et al. 2023) without running untrusted pickle code.

Files (https://dl.fbaipublicfiles.com/gismo/): {train,val,test}_comments_subs.pkl and vocab_ingrs.pkl.
Each split is a list of dicts: {"id": recipe_id, "ingredients": [[name, synonym...], ...],
"subs": (source_name, target_name)}. The vocabulary is an inv_cooking ``Vocabulary``:
``word2idx`` maps every surface name to a cluster index, ``idx2word[i]`` is the list of
names of that cluster, and its first element is the canonical name GISMo uses.
"""

from __future__ import annotations

import csv
import io
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path


def normalize(name: str) -> str:
    """Shared ingredient key: lower case, spaces and hyphens -> '_' ("Olive Oil" -> "olive_oil")."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


class Vocabulary:
    """Stand-in for inv_cooking.datasets.vocabulary.Vocabulary, filled by unpickling."""

    def __init__(self):
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self.idx = 0

    def canonical(self, name: str) -> str:
        """Canonical GISMo name of ``name`` (the first name of its cluster)."""
        if name not in self.word2idx:
            return name
        names = self.idx2word[self.word2idx[name]]
        return names[0] if isinstance(names, (list, tuple)) else names

    def canonical_names(self) -> list[str]:
        out = []
        for names in self.idx2word.values():
            name = names[0] if isinstance(names, (list, tuple)) else names
            if not (name.startswith("<") and name.endswith(">")):  # special tokens such as <end>
                out.append(name)
        return sorted(set(out))


_SAFE = {
    ("builtins", n) for n in ("dict", "list", "tuple", "set", "frozenset", "str", "int", "float", "bool")
} | {("collections", "OrderedDict"), ("collections", "defaultdict")}


class _SafeUnpickler(pickle.Unpickler):
    """Only plain containers and the Vocabulary class: a malicious pickle cannot run code."""

    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        if (module, name) in _SAFE:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the pickle")


def safe_load(path: str | Path):
    with open(path, "rb") as f:
        return _SafeUnpickler(io.BufferedReader(f)).load()


@dataclass(frozen=True)
class SubPair:
    recipe_id: str
    source: str  # normalized canonical name
    target: str
    context: tuple[str, ...] = field(default=(), compare=False)


def load_vocab(data_dir: str | Path) -> Vocabulary:
    vocab = safe_load(Path(data_dir) / "vocab_ingrs.pkl")
    if not isinstance(vocab, Vocabulary):
        raise TypeError(f"vocab_ingrs.pkl holds a {type(vocab).__name__}, expected a Vocabulary")
    return vocab


def load_split(data_dir: str | Path, split: str, vocab: Vocabulary | None = None) -> list[SubPair]:
    """Read ``<split>_comments_subs.pkl`` into SubPairs, names mapped to canonical + normalized."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    vocab = vocab or load_vocab(data_dir)
    rows = safe_load(Path(data_dir) / f"{split}_comments_subs.pkl")
    pairs = []
    for row in rows:
        src, tgt = row["subs"][0], row["subs"][1]
        ctx = []
        for ing in row.get("ingredients", []):
            name = ing[0] if isinstance(ing, (list, tuple)) else ing
            ctx.append(normalize(vocab.canonical(name)))
        pairs.append(
            SubPair(
                recipe_id=str(row["id"]),
                source=normalize(vocab.canonical(src)),
                target=normalize(vocab.canonical(tgt)),
                context=tuple(ctx),
            )
        )
    return pairs


def load_flavorgraph_ingredients(nodes_csv: str | Path) -> list[str]:
    """GISMo's candidate set: the ingredient nodes of FlavorGraph (nodes_191120.csv)."""
    with open(nodes_csv, newline="", encoding="utf-8") as f:
        return sorted({normalize(r["name"]) for r in csv.DictReader(f) if "ingredient" in r["node_type"]})


def recipes_from_pairs(pairs: list[SubPair]) -> list[tuple[str, ...]]:
    """One ingredient list per distinct recipe (several pairs can share a recipe)."""
    seen = {}
    for p in pairs:
        seen.setdefault(p.recipe_id, p.context)
    return list(seen.values())
