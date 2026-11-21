#!/usr/bin/env bash
# Downloads the recipe datasets of the project into ./data (not tracked by git).
#   - Open Recipes snapshot 2017-01-07 (173,278 recipes, CC BY 3.0): the training corpus.
#   - Recipe1MSubs (GISMo, CC-BY-NC): ingredient vocabulary and substitution pairs.
#     Evaluation only: never train on it.
# The original Open Recipes bucket answers 403, so the snapshot comes from this repository's
# GitHub release. To use a local copy instead:
#   OPENRECIPES_GZ=/path/to/20170107-061401-recipeitems.json.gz ./download_recipes.sh
set -euo pipefail

DATA="${1:-data}"
mkdir -p "$DATA/openrecipes" "$DATA/recipe1msubs"
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

# Every file goes to <name>.part first and only gets its real name once the checksum matches,
# so a wrong copy or a cut download never stays behind to break the next run.
checked() { echo "$1  $2" | $SHA -c - >/dev/null 2>&1; }
fetch() {  # fetch <file> <sha256> <command...>; the command writes to the path given last
  local f="$1" sum="$2"; shift 2
  if [ -f "$f" ]; then
    if checked "$sum" "$f"; then echo "$f: OK"; return; fi
    echo "$f: checksum mismatch, fetching it again" >&2
    rm -f "$f"
  fi
  rm -f "$f.part"
  if ! "$@" "$f.part"; then rm -f "$f.part"; return 1; fi
  if ! checked "$sum" "$f.part"; then
    rm -f "$f.part"
    echo "$f: checksum mismatch, the file was not kept" >&2
    return 1
  fi
  mv "$f.part" "$f"
  echo "$f: OK"
}

OR_FILE="$DATA/openrecipes/20170107-061401-recipeitems.json.gz"
OR_SHA="6e40a82e0bd83468f8020b461924d9d6cf8bdba822642c5276868104acb6d682"
or_copy() { cp "$OPENRECIPES_GZ" "$1"; }
OR_URL=https://github.com/AlejandroB10/pa164-levy2015-reproduction/releases/download/openrecipes-2017-01-07/20170107-061401-recipeitems.json.gz
or_curl() { curl -fL -o "$1" "$OR_URL"; }
if [ -n "${OPENRECIPES_GZ:-}" ]; then OR_GET=or_copy; else OR_GET=or_curl; fi
if ! fetch "$OR_FILE" "$OR_SHA" "$OR_GET"; then
  echo "Open Recipes download failed. Fetch $OR_URL by hand and set OPENRECIPES_GZ=<path>." >&2
  exit 1
fi

GISMO=https://dl.fbaipublicfiles.com/gismo
gismo_get() { curl -fL -o "$1" "$GISMO/$name"; }
while read -r sum name; do
  fetch "$DATA/recipe1msubs/$name" "$sum" gismo_get
done <<'EOF'
ad8c62dee7d1314be6e173c3f3bbde473c2c79352bd6e3ccf656a6ac9a488ed4 vocab_ingrs.pkl
b4d58e6c6c6747671e531812393397acf75c02d2a9ce00c2edcc7794bf567c81 train_comments_subs.pkl
4d7e96837c051fbec7464346638942cb66b7841cb998dd6abe1382e2f2ddb3dd val_comments_subs.pkl
0758382cae59697958dd2a5e509c15f8d0417ea1a9e079fd75792779f1ce6174 test_comments_subs.pkl
EOF
