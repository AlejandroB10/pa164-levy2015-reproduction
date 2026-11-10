import pytest

from recipes.normalize import Normalizer

# A small slice of the Recipe1MSubs vocabulary: surface form -> canonical ingredient.
SURFACE = {
    "garlic": "garlic",
    "garlic_clove": "garlic_clove",
    "garlic_cloves": "garlic_clove",
    "clove": "clove",
    "cloves": "clove",
    "ground_clove": "ground_clove",
    "ground_cloves": "ground_clove",
    "cheese": "cheese",
    "butter": "butter",
    "stick": "stick",
    "worcestershire_sauce": "worcestershire_sauce",
    "biscuit": "biscuit",
    "biscuits": "biscuit",
    "gravy": "gravy",
    "sausage_gravy": "gravy",
    "salt": "salt",
    "salt_and_pepper": "salt_and_pepper",
    "all_purpose_flour": "all_purpose_flour",
    "flour": "flour",
    "tomato": "tomato",
    "tomatoes": "tomato",
    "olive_oil": "olive_oil",
    "oil": "oil",
    "red_chili": "red_chilies",
    "creme_fraiche": "creme_fraiche",
    "parmesan_cheese": "parmesan_cheese",
    "gelatin": "gelatin",
    "barbecue_sauce": "barbecue_sauce",
}


@pytest.fixture
def norm():
    return Normalizer(SURFACE)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("2 cloves garlic, minced", "garlic"),
        ("1 tsp ground cloves", "ground_clove"),
        ("3 garlic cloves", "garlic_clove"),
        ("1 pound Cheese (Provolone, Swiss, Mozzarella, Even Cheez Whiz!)", "cheese"),
        ("1-1/2 stick (3/4 Cup) Cold Butter, Cut Into Pieces", "butter"),
        ("3 cups All-purpose Flour", "all_purpose_flour"),
        ("1 (14 oz) can diced tomatoes", "tomato"),
        ("Salt and pepper to taste", "salt_and_pepper"),
        ("2 Tbsp. extra virgin olive oil", "olive_oil"),
        ("½ cup oil", "oil"),
        ("1 red chilli , deseeded and finely chopped", "red_chilies"),
        ("2 tbsp crème fraîche", "creme_fraiche"),
        ("50g parmesan cheese, grated", "parmesan_cheese"),
        ("2 sheets gelatine", "gelatin"),
        ("1/4 cup barbeque sauce", "barbecue_sauce"),
    ],
)
def test_normalizes_ingredient_lines(norm, line, expected):
    assert norm.normalize(line) == expected


def test_unit_words_are_not_ingredients(norm):
    # "dash" must never be read as the ingredient; worcestershire alone is not in the vocab.
    assert norm.normalize("Dash Of Worcestershire") is None


def test_section_headers_are_skipped(norm):
    assert norm.normalize(" SAUSAGE GRAVY") is None
    assert norm.normalize("For the gravy:") is None


def test_ingredient_word_after_comma_is_used_when_head_has_none(norm):
    assert norm.normalize("1 cup shredded, cheese") == "cheese"


def test_quantity_only_lines_are_recognised(norm):
    # tastykitchen pages in the dump lost the ingredient names and keep only the amounts.
    assert norm.is_quantity_only("1-½ cup 1-½ cup")
    assert norm.is_quantity_only("8 ounces, weight 8 ounces, weight")
    assert norm.is_quantity_only("2 whole 2 whole")
    assert not norm.is_quantity_only("2 cloves garlic")
    assert not norm.is_quantity_only("1 cup whole milk")


def test_unknown_and_empty_lines(norm):
    assert norm.normalize("1 cup unobtainium") is None
    assert norm.normalize("   ") is None
