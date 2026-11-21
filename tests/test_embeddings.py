import numpy as np
import pytest

from recipes.embeddings import load_embeddings, save_embeddings


def test_round_trip_keeps_names_order_and_values(tmp_path):
    names = ["garlic", "olive_oil", "creme_fraiche"]
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    path = tmp_path / "sgns.txt"
    save_embeddings(path, names, vectors)
    got_names, got_vectors = load_embeddings(path)
    assert got_names == names
    np.testing.assert_allclose(got_vectors, vectors, rtol=1e-6)
    assert path.read_text().splitlines()[0] == "3 4"  # word2vec text header: rows, dims


def test_gensim_reads_the_same_file(tmp_path):
    from gensim.models import KeyedVectors

    path = tmp_path / "v.txt"
    save_embeddings(path, ["a", "b"], np.eye(2, dtype=np.float32))
    kv = KeyedVectors.load_word2vec_format(str(path))
    assert kv.index_to_key == ["a", "b"]


@pytest.mark.parametrize("bad", ["two words", "tab\tname", ""])
def test_names_must_be_single_tokens(tmp_path, bad):
    with pytest.raises(ValueError):
        save_embeddings(tmp_path / "v.txt", [bad], np.zeros((1, 2), dtype=np.float32))


def test_rows_must_match_names(tmp_path):
    with pytest.raises(ValueError):
        save_embeddings(tmp_path / "v.txt", ["a", "b"], np.zeros((1, 2), dtype=np.float32))


def test_truncated_file_is_rejected(tmp_path):
    # A file cut short must fail loudly instead of returning uninitialised rows.
    path = tmp_path / "v.txt"
    save_embeddings(path, ["a", "b", "c"], np.ones((3, 2), dtype=np.float32))
    path.write_text("\n".join(path.read_text().splitlines()[:3]) + "\n")
    with pytest.raises(ValueError, match="rows"):
        load_embeddings(path)


def test_row_with_wrong_width_is_rejected(tmp_path):
    path = tmp_path / "v.txt"
    path.write_text("2 2\na 1 2\nb 1\n")
    with pytest.raises(ValueError, match="b"):
        load_embeddings(path)
