import numpy as np
import pytest

from recipes.sgns import read_recipes, train_sgns, vectors

CORPUS = ["butter flour sugar egg", "margarine flour sugar egg", "garlic olive_oil onion",
          "garlic olive_oil tomato onion basil"] * 20


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "ingredients.txt"
    path.write_text("\n".join(CORPUS) + "\n")
    return path


def test_each_line_is_one_recipe(corpus):
    recipes = read_recipes(corpus)
    assert len(recipes) == len(CORPUS)
    assert recipes[0] == ["butter", "flour", "sugar", "egg"]


def test_window_covers_the_longest_recipe_and_does_not_shrink(corpus):
    model = train_sgns(corpus, vector_size=8, epochs=1, min_count=1)
    assert model.window == 5  # longest recipe has 5 ingredients
    assert model.shrink_windows is False
    assert model.sg == 1


def test_hyperparameters_reach_gensim(corpus):
    model = train_sgns(corpus, vector_size=8, epochs=2, min_count=1, negative=3,
                       ns_exponent=1.0, sample=0)
    assert (model.vector_size, model.epochs, model.negative) == (8, 2, 3)
    assert model.ns_exponent == 1.0 and model.sample == 0
    assert set(model.wv.index_to_key) == {w for line in CORPUS for w in line.split()}


def test_w_plus_c_adds_the_aligned_context_vectors(corpus):
    model = train_sgns(corpus, vector_size=8, epochs=1, min_count=1)
    names, w = vectors(model)
    names_wc, wc = vectors(model, w_plus_c=True)
    assert names == names_wc == list(model.wv.index_to_key)
    assert model.syn1neg.shape == model.wv.vectors.shape
    np.testing.assert_allclose(wc, model.wv.vectors + model.syn1neg, rtol=1e-6)
    np.testing.assert_allclose(w, model.wv.vectors)
