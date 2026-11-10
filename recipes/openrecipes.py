"""Read the Open Recipes dump (one JSON object per line, gzipped) as ingredient lists."""
import gzip
import json


def iter_ingredient_lists(path):
    """Ingredient lines of each recipe, skipping empty recipes and duplicates.

    The dump was scraped from many sites and repeats recipes, so a recipe is dropped when its
    url or its exact ingredient list was already seen.
    """
    seen_urls, seen_lists = set(), set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in f:
            recipe = json.loads(row)
            lines = [line.strip() for line in (recipe.get("ingredients") or "").split("\n") if line.strip()]
            if not lines:
                continue
            url, key = recipe.get("url"), "\n".join(lines).lower()
            if url in seen_urls or key in seen_lists:
                continue
            if url:
                seen_urls.add(url)
            seen_lists.add(key)
            yield lines


def recipe_tokens(lines, normalizer):
    """Canonical ingredients of one recipe (each once, in first-seen order) and its unmatched lines."""
    tokens, unmatched = [], []
    for line in lines:
        if normalizer.is_header(line):
            continue
        canon = normalizer.normalize(line)
        if canon is None:
            unmatched.append(line)
        elif canon not in tokens:
            tokens.append(canon)
    return tokens, unmatched
