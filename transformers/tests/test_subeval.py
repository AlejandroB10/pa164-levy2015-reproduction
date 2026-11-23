"""Run with: python -m unittest discover -s tests  (from the pa164-recipes-eval folder)."""

import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subeval.data import SubPair, load_split, load_vocab, normalize, safe_load  # noqa: E402
from subeval.evaluate import evaluate, rank_pairs  # noqa: E402
from subeval.metrics import rank_of, summarize  # noqa: E402
from subeval.scorers import (  # noqa: E402
    EmbeddingScorer, FrequencyScorer, GismoTableScorer, PairFrequencyScorer, PMIScorer, load_vectors,
)

# A fake inv_cooking module, so the test pickles look like GISMo's files.
_mod = types.ModuleType("inv_cooking.datasets.vocabulary")


class Vocabulary:  # same shape as inv_cooking's class
    def __init__(self):
        self.word2idx, self.idx2word, self.idx = {}, {}, 0

    def add(self, names):
        for n in names:
            self.word2idx[n] = self.idx
        self.idx2word[self.idx] = list(names)
        self.idx += 1


Vocabulary.__module__ = _mod.__name__
_mod.Vocabulary = Vocabulary
for name in ("inv_cooking", "inv_cooking.datasets", "inv_cooking.datasets.vocabulary"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["inv_cooking.datasets.vocabulary"] = _mod


def write_fake_data(d: Path):
    v = Vocabulary()
    for names in (["butter", "unsalted_butter"], ["margarine"], ["olive_oil"], ["oil", "vegetable_oil"],
                  ["milk"], ["soy_milk"], ["flour"], ["sugar"]):
        v.add(names)
    pickle.dump(v, open(d / "vocab_ingrs.pkl", "wb"))
    ing = [["flour"], ["sugar"], ["unsalted_butter"], ["milk"]]
    train = [
        {"id": "r1", "ingredients": ing, "subs": ("unsalted_butter", "margarine")},
        {"id": "r2", "ingredients": ing, "subs": ("butter", "margarine")},
        {"id": "r3", "ingredients": [["olive_oil"], ["flour"]], "subs": ("olive_oil", "vegetable_oil")},
        {"id": "r4", "ingredients": [["milk"], ["sugar"]], "subs": ("milk", "soy_milk")},
    ]
    test = [
        {"id": "t1", "ingredients": ing, "subs": ("butter", "margarine")},
        {"id": "t1", "ingredients": ing, "subs": ("butter", "oil")},  # second gold target, same recipe
        {"id": "t2", "ingredients": [["milk"]], "subs": ("milk", "soy_milk")},
    ]
    for split, rows in (("train", train), ("val", test), ("test", test)):
        pickle.dump(rows, open(d / f"{split}_comments_subs.pkl", "wb"))


class TestMetrics(unittest.TestCase):
    def test_rank_basic_and_ties(self):
        self.assertEqual(rank_of([0.1, 0.9, 0.5], 1), 1)
        self.assertEqual(rank_of([0.1, 0.9, 0.5], 0), 3)
        # ties count against the target: a constant scorer gets the worst rank
        self.assertEqual(rank_of([1.0, 1.0, 1.0, 1.0], 2), 4)

    def test_rank_exclude(self):
        self.assertEqual(rank_of([0.9, 0.8, 0.1], 1, exclude=[0]), 1)

    def test_summarize(self):
        m = summarize([1, 2, 0])  # 0 = not covered -> miss
        self.assertAlmostEqual(m["mrr"], (1 + 0.5 + 0) / 3 * 100)
        self.assertAlmostEqual(m["hit@1"], 100 / 3)
        self.assertAlmostEqual(m["hit@3"], 200 / 3)


class TestData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_fake_data(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_names(self):
        pairs = load_split(self.d, "train")
        self.assertEqual(pairs[0].source, "butter")  # unsalted_butter -> cluster head
        self.assertEqual(pairs[2].target, "oil")  # vegetable_oil -> oil
        self.assertEqual(pairs[0].context[2], "butter")

    def test_safe_unpickler_refuses_code(self):
        class Evil:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        p = self.d / "evil.pkl"
        pickle.dump(Evil(), open(p, "wb"))
        with self.assertRaises(pickle.UnpicklingError):
            safe_load(p)

    def test_end_to_end_cli(self):
        vec = self.d / "vec.txt"
        vec.write_text(
            "8 2\nbutter 1 0\nmargarine 0.9 0.1\noil 0.7 0.3\nolive_oil 0.6 0.4\n"
            "milk 0 1\nsoy_milk 0.1 0.9\nflour -1 0\nsugar -1 -0.2\n", encoding="utf-8")
        out = self.d / "res.csv"
        r = subprocess.run(
            [sys.executable, str(ROOT / "run_eval.py"), "--data-dir", str(self.d), "--split", "val",
             "--model", f"toy={vec}", "--out", str(out)],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("toy", out.read_text())


class TestScorers(unittest.TestCase):
    def setUp(self):
        self.words = ["butter", "margarine", "oil", "milk", "soy_milk"]
        self.vecs = np.array([[1, 0], [0.9, 0.1], [0.6, 0.4], [0, 1], [0.1, 0.9]], dtype=np.float32)
        self.cands = list(self.words)
        self.test = [
            SubPair("t1", "butter", "margarine"),
            SubPair("t1", "butter", "oil"),
            SubPair("t2", "milk", "soy_milk"),
        ]

    def test_embedding_ranks_and_source_excluded(self):
        sc = EmbeddingScorer(self.words, self.vecs)
        # candidates for butter, without butter: margarine > oil > soy_milk > milk
        self.assertEqual(rank_pairs(sc, self.test, self.cands), [1, 2, 1])

    def test_filtered_removes_other_gold(self):
        sc = EmbeddingScorer(self.words, self.vecs)
        self.assertEqual(rank_pairs(sc, self.test, self.cands, filtered=True), [1, 1, 1])

    def test_load_vectors_formats(self):
        with tempfile.TemporaryDirectory() as t:
            txt = Path(t) / "v.txt"
            txt.write_text("2 2\nOlive-Oil 1 2\nolive_oil 9 9\n", encoding="utf-8")
            w, m = load_vectors(txt)
            self.assertEqual(w, ["olive_oil"])  # normalized, first one kept
            np.testing.assert_allclose(m[0], [1, 2])
            npz = Path(t) / "v.npz"
            np.savez(npz, words=np.array(["a", "b"]), vectors=np.eye(2))
            w, m = load_vectors(npz)
            self.assertEqual(w, ["a", "b"])

    def test_zero_vector_not_covered(self):
        sc = EmbeddingScorer(["a", "b"], np.array([[1.0, 0], [0, 0]]))
        self.assertEqual(sc.vocab, {"a"})

    def test_baselines(self):
        train = [SubPair("r", "butter", "margarine"), SubPair("r2", "butter", "margarine"),
                 SubPair("r3", "milk", "soy_milk"), SubPair("r4", "oil", "butter")]
        self.assertEqual(rank_pairs(FrequencyScorer(train), self.test, self.cands)[0], 1)
        pf = PairFrequencyScorer(train)
        # butter -> oil: soy_milk (popularity 1) above, milk tied at 0 -> pessimistic rank 4
        self.assertEqual(rank_pairs(pf, self.test, self.cands), [1, 4, 1])
        pmi = PMIScorer([["butter", "flour"], ["butter", "margarine"], ["milk", "soy_milk"]])
        self.assertGreater(pmi.pmi("milk", "soy_milk"), pmi.pmi("butter", "flour") - 1)
        self.assertEqual(pmi.pmi("milk", "flour"), float("-inf"))

    def test_gismo_baselines_and_keep_source(self):
        train = [SubPair("r", "butter", "margarine"), SubPair("r2", "butter", "margarine"),
                 SubPair("r3", "butter", "oil"), SubPair("r4", "milk", "soy_milk")]
        freq = GismoTableScorer(train, "gismo-freq")  # counts sources, as GISMo's code does
        np.testing.assert_array_equal(freq.scores("milk", ["butter", "milk", "oil"]), [3, 1, 0])
        lt = GismoTableScorer(train, "gismo-ltfreq")
        np.testing.assert_array_equal(lt.scores("butter", ["margarine", "oil", "milk"]), [2, 1, 0])
        # keep_source: the source (butter, top source count) now outranks margarine
        p = [SubPair("t", "milk", "butter")]
        self.assertEqual(rank_pairs(freq, p, ["butter", "milk", "oil"], keep_source=True), [1])
        p = [SubPair("t", "butter", "milk")]
        self.assertEqual(rank_pairs(freq, p, ["butter", "milk", "oil"], keep_source=True), [2])
        self.assertEqual(rank_pairs(freq, p, ["butter", "milk", "oil"]), [1])

    def test_combined_backoff_and_prior(self):
        from subeval.combine import CombinedScorer, LogFreq, LogPairCount, _z
        np.testing.assert_allclose(_z(np.array([1.0, -np.inf, 3.0])).mean(), 0, atol=1e-12)
        train = [SubPair("r", "butter", "oil"), SubPair("r2", "milk", "soy_milk"), SubPair("r3", "milk", "soy_milk")]
        emb = EmbeddingScorer(self.words, self.vecs)
        # back-off: the seen pair butter->oil comes first even though margarine is closer
        sc = CombinedScorer([LogPairCount(train), emb], [100, 1], raw=[0])
        self.assertEqual(rank_pairs(sc, [SubPair("t", "butter", "oil")], self.cands), [1])
        # a strong popularity prior lifts the popular soy_milk above margarine for butter
        sc = CombinedScorer([emb, LogFreq(train)], [1, 10])
        self.assertEqual(rank_pairs(sc, [SubPair("t", "butter", "soy_milk")], self.cands), [1])

    def test_common_mode_same_candidates(self):
        a = EmbeddingScorer(self.words, self.vecs, "a")
        b = EmbeddingScorer(self.words[:4], self.vecs[:4], "b")  # no soy_milk
        rows, pairs, _ = evaluate([a, b], self.test, self.cands, mode="common")
        self.assertEqual(len(pairs), 2)  # milk->soy_milk dropped for everyone
        self.assertEqual({r["n_candidates"] for r in rows}, {4})
        rows, _, per = evaluate([a, b], self.test, self.cands, mode="own")
        self.assertEqual(per["b"][2], 0)  # b cannot rank soy_milk: counted as a miss
        self.assertLess(rows[1]["mrr"], rows[0]["mrr"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
