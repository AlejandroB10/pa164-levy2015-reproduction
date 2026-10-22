#!/usr/bin/env bash
# Downloads every dataset the experiment needs into ./data (not tracked by git).
# Run inside the conda env (it reads SimLex-999 from gensim's bundled test data).
set -euo pipefail

DATA="${1:-data}"
mkdir -p "$DATA/text8" "$DATA/eval-similarity"

# text8: first 10^8 bytes of English Wikipedia (Matt Mahoney), 17,005,207 tokens.
if [ ! -f "$DATA/text8/text8" ]; then
  curl -fL -o "$DATA/text8/text8.zip" http://mattmahoney.net/dc/text8.zip
  unzip -o "$DATA/text8/text8.zip" -d "$DATA/text8"
fi

# Google analogy questions (word2vec repository, Apache-2.0).
curl -fL -o "$DATA/text8/questions-words.txt" \
  https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt

# Similarity sets as packaged by the paper's authors (BIU-NLP/hyperwords, testsets/ws).
HW=https://raw.githubusercontent.com/BIU-NLP/hyperwords/master/testsets/ws
for f in ws353_similarity.txt ws353_relatedness.txt bruni_men.txt radinsky_mturk.txt luong_rare.txt; do
  curl -fL -o "$DATA/eval-similarity/$f" "$HW/$f"
done

# SimLex-999 ships with gensim.
python -c "import shutil, sys; from gensim.test.utils import datapath; shutil.copy(datapath('simlex999.txt'), sys.argv[1])" \
  "$DATA/eval-similarity/simlex999_gensim.txt"

wc -w "$DATA/text8/text8"
