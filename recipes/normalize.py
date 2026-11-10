"""Map a free-text ingredient line ("2 cloves garlic, minced") to one canonical ingredient."""
import re
import unicodedata

# Units and containers. They are dropped only right after a quantity, because some of them
# ("clove", "stick", "jar") are also ingredients in the vocabulary.
UNITS = {
    "c", "cup", "cups", "tsp", "tsps", "teaspoon", "teaspoons", "tbsp", "tbsps", "tbs", "tablespoon",
    "tablespoons", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gram", "grams", "kg",
    "ml", "l", "liter", "liters", "litre", "litres", "qt", "quart", "quarts", "pt", "pint", "pints",
    "gallon", "gallons", "clove", "cloves", "stick", "sticks", "can", "cans", "jar", "jars", "package",
    "packages", "pkg", "box", "boxes", "bag", "bags", "bottle", "bottles", "bunch", "bunches", "head",
    "heads", "sprig", "sprigs", "slice", "slices", "piece", "pieces", "pinch", "pinches", "dash",
    "dashes", "handful", "handfuls", "container", "containers", "envelope", "envelopes", "sheet", "sheets", "of", "weight", "fl", "fluid",
}
# Size and container words that are part of some ingredient names ("whole milk", "bay leaves"),
# so they are never stripped; a line made only of them and amounts has no ingredient name.
COUNT_WORDS = {
    "whole", "large", "medium", "small", "stalk", "stalks", "loaf", "loaves", "block", "blocks", "tub",
    "tubs", "cube", "cubes", "leaf", "leaves", "drop", "drops", "bulb", "bulbs", "carton", "cartons",
    "strip", "strips", "ear", "ears",
}
# Never an ingredient on their own, wherever they appear.
STOP = {"dash", "dashes", "pinch", "pinches", "of", "to", "taste", "or", "and", "a", "an", "the"}

QUANTITY = re.compile(r"^[\d\s/.\-–½⅓⅔¼¾⅛]+")
PARENS = re.compile(r"\([^)]*\)?|\[[^\]]*\]?")
WORD = re.compile(r"[a-z0-9%&]+")
# British and variant spellings that the (American) Recipe1M vocabulary does not list.
SPELLING = {"chilli": "chili", "chillies": "chilies", "gelatine": "gelatin", "barbeque": "barbecue"}


class Normalizer:
    def __init__(self, surface_to_canonical):
        self.forms = {tuple(form.split("_")): canon for form, canon in surface_to_canonical.items()}
        self.max_len = max(map(len, self.forms))

    def normalize(self, line):
        """Canonical ingredient of `line`, or None for headers and unmatched lines."""
        if self.is_header(line):
            return None
        text = unicodedata.normalize("NFKD", line.strip().lower())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))  # crème -> creme
        text = PARENS.sub(" ", text).replace("'", "").replace("-", " ")
        head = text.split(",")[0]
        return self._match(head) or self._match(text.replace(",", " "))

    @staticmethod
    def is_header(line):
        """Blank lines and section titles ("For the gravy:", "SAUSAGE GRAVY")."""
        text = line.strip()
        return not text or text.endswith(":") or (text.isupper() and not any(ch.isdigit() for ch in text))

    def is_quantity_only(self, line):
        """A line with no ingredient name at all ("1 cup 1 cup", a broken scrape)."""
        text = PARENS.sub(" ", line.lower()).replace(",", " ").replace("-", " ")
        return all(t.isdigit() or t in UNITS or t in COUNT_WORDS for t in WORD.findall(text))

    def _match(self, text):
        words = [SPELLING.get(w, w) for w in WORD.findall(QUANTITY.sub("", text))]
        tokens = self._strip_quantity(words)
        best = None  # (length, start); longest wins, ties go to the rightmost span (the head noun)
        for start in range(len(tokens)):
            for n in range(min(self.max_len, len(tokens) - start), 0, -1):
                span = tuple(tokens[start : start + n])
                if span in self.forms and not (n == 1 and span[0] in STOP):
                    if best is None or (n, start) >= best:
                        best = (n, start)
                    break
        if best is None:
            return None
        n, start = best
        return self.forms[tuple(tokens[start : start + n])]

    @staticmethod
    def _strip_quantity(tokens):
        """Drop leading numbers and units ("1 1 2 stick", "14 oz can")."""
        i = 0
        while i < len(tokens) and (tokens[i].isdigit() or tokens[i] in UNITS):
            i += 1
        return tokens[i:]
