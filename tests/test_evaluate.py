import numpy as np
import pytest

from recipes.evaluate import evaluate_substitutions

# 2-d vectors: a is closest to b, then c, then d (cosine).
NAMES = ["a", "b", "c", "d"]
VECTORS = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [-1.0, 0.0]], dtype=np.float32)


def test_ranks_by_cosine_and_excludes_the_target():
    m = evaluate_substitutions(NAMES, VECTORS, [("a", "b"), ("a", "c"), ("a", "d")])
    # ranks among {b, c, d}: 1, 2, 3
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@3"] == pytest.approx(1.0)
    assert m["coverage"] == 1.0 and m["n"] == 3


def test_uncovered_pairs_count_as_zero_but_covered_mrr_ignores_them():
    m = evaluate_substitutions(NAMES, VECTORS, [("a", "b"), ("a", "zzz"), ("yyy", "a")])
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["mrr_covered"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1 / 3)
    assert m["hit@10"] == pytest.approx(1 / 3)


def test_scale_of_vectors_does_not_matter():
    m1 = evaluate_substitutions(NAMES, VECTORS, [("a", "c")])
    m2 = evaluate_substitutions(NAMES, VECTORS * np.array([[10], [1], [3], [7]]), [("a", "c")])
    assert m1 == m2


def test_duplicate_instances_count_each_time():
    m = evaluate_substitutions(NAMES, VECTORS, [("a", "b"), ("a", "b"), ("a", "d")])
    assert m["mrr"] == pytest.approx((1 + 1 + 1 / 3) / 3)


def test_zero_or_nan_vector_counts_as_not_covered():
    # A count model can leave an all-zero row for a rare ingredient; it must not get rank 1.
    names = ["a", "b", "c", "z", "n"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 0.0], [np.nan, 1.0]])
    m = evaluate_substitutions(names, vectors, [("z", "a"), ("n", "a"), ("a", "z"), ("a", "b")])
    assert m["coverage"] == pytest.approx(1 / 4)
    assert m["mrr"] == pytest.approx(1 / 4)  # only (a, b) is scored, at rank 1


def test_ties_count_against_the_model():
    # b and c are equally close to a: the substitute shares rank 2, it does not win rank 1.
    names = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    m = evaluate_substitutions(names, vectors, [("a", "b")])
    assert m["mrr"] == pytest.approx(1 / 2)


def test_names_must_match_rows_and_be_unique():
    with pytest.raises(ValueError):
        evaluate_substitutions(["a", "b"], VECTORS, [("a", "b")])
    with pytest.raises(ValueError):
        evaluate_substitutions(["a", "b", "b", "d"], VECTORS, [("a", "b")])


def test_no_pairs_is_an_error():
    with pytest.raises(ValueError):
        evaluate_substitutions(NAMES, VECTORS, [])
