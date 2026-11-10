import gzip
import json

from recipes.normalize import Normalizer
from recipes.openrecipes import iter_ingredient_lists, recipe_tokens


def write_dump(path, recipes):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recipes:
            f.write(json.dumps(r) + "\n")


def test_duplicates_and_empty_recipes_are_dropped(tmp_path):
    dump = tmp_path / "dump.json.gz"
    write_dump(dump, [
        {"name": "A", "url": "http://x/a", "ingredients": "1 egg\n2 cups flour"},
        {"name": "A again", "url": "http://x/a", "ingredients": "1 egg"},          # same url
        {"name": "B", "url": "http://x/b", "ingredients": "1 egg\n2 cups flour"},  # same list
        {"name": "C", "url": "http://x/c", "ingredients": ""},
        {"name": "D", "url": "http://x/d"},
        {"name": "E", "url": "http://x/e", "ingredients": "salt\n"},
    ])
    lists = list(iter_ingredient_lists(dump))
    assert lists == [["1 egg", "2 cups flour"], ["salt"]]


def test_recipe_tokens_are_unique_in_first_seen_order():
    norm = Normalizer({"egg": "egg", "eggs": "egg", "flour": "flour", "salt": "salt"})
    lines = ["2 eggs", "1 cup flour", "SAUCE", "1 egg yolk", "a pinch of nothing", "salt"]
    assert recipe_tokens(lines, norm) == (["egg", "flour", "salt"], ["a pinch of nothing"])
