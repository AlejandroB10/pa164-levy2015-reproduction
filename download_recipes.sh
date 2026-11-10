#!/usr/bin/env bash
# Downloads the recipe datasets of the project into ./data (not tracked by git).
#   - Open Recipes snapshot 2017-01-07 (173,278 recipes, CC BY 3.0): the training corpus.
#   - Recipe1MSubs (GISMo, CC-BY-NC): ingredient vocabulary and substitution pairs.
#     Evaluation only: never train on it.
# The Open Recipes bucket now answers 403. If the download fails, pass a local copy:
#   OPENRECIPES_GZ=/path/to/20170107-061401-recipeitems.json.gz ./download_recipes.sh
set -euo pipefail

DATA="${1:-data}"
mkdir -p "$DATA/openrecipes" "$DATA/recipe1msubs"
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

OR_FILE="$DATA/openrecipes/20170107-061401-recipeitems.json.gz"
OR_SHA="6e40a82e0bd83468f8020b461924d9d6cf8bdba822642c5276868104acb6d682"
if [ ! -f "$OR_FILE" ]; then
  if [ -n "${OPENRECIPES_GZ:-}" ]; then
    cp "$OPENRECIPES_GZ" "$OR_FILE"
  elif ! curl -fL -o "$OR_FILE" https://s3.amazonaws.com/openrecipes/20170107-061401-recipeitems.json.gz; then
    rm -f "$OR_FILE"
    echo "Open Recipes download failed. Ask the team for the file and set OPENRECIPES_GZ=<path>." >&2
    exit 1
  fi
fi
echo "$OR_SHA  $OR_FILE" | $SHA -c -

GISMO=https://dl.fbaipublicfiles.com/gismo
while read -r sum name; do
  f="$DATA/recipe1msubs/$name"
  [ -f "$f" ] || curl -fL -o "$f" "$GISMO/$name"
  echo "$sum  $f" | $SHA -c -
done <<'EOF'
ad8c62dee7d1314be6e173c3f3bbde473c2c79352bd6e3ccf656a6ac9a488ed4 vocab_ingrs.pkl
b4d58e6c6c6747671e531812393397acf75c02d2a9ce00c2edcc7794bf567c81 train_comments_subs.pkl
4d7e96837c051fbec7464346638942cb66b7841cb998dd6abe1382e2f2ddb3dd val_comments_subs.pkl
0758382cae59697958dd2a5e509c15f8d0417ea1a9e079fd75792779f1ce6174 test_comments_subs.pkl
EOF
