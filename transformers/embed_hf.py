"""Context-free ingredient embeddings from Hugging Face models, in the shared .npz format.

Each ingredient name is encoded on its own ("olive_oil" -> "olive oil", optionally inside a
template), so the vector does not depend on the recipe, like SGNS / PPMI / SVD.

    # sentence-embedding model (BGE: [CLS] pooling, as in its 1_Pooling config)
    python embed_hf.py --model BAAI/bge-base-en-v1.5 --pooling cls --out vectors/bge.npz

    # averaged BERT: mean of the word-piece vectors of the name, last layer or last 4
    python embed_hf.py --model bert-base-cased --pooling mean --layers -4 -3 -2 -1 --out vectors/bert_avg.npz

Names come from FlavorGraph (--candidates) or from vocab_ingrs.pkl (--data-dir).
Runs offline from the Hugging Face cache by default (set --online to allow downloads).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("USE_TF", "0")  # transformers otherwise imports TensorFlow (slow, noisy)

import numpy as np  # noqa: E402

from subeval.data import load_flavorgraph_ingredients, load_vocab, normalize  # noqa: E402


def ingredient_names(args) -> list[str]:
    if args.candidates:
        return load_flavorgraph_ingredients(args.candidates)
    if args.data_dir:
        return sorted({normalize(n) for n in load_vocab(args.data_dir).canonical_names()})
    raise SystemExit("give --candidates (FlavorGraph nodes csv) or --data-dir (Recipe1MSubs)")


def to_text(name: str) -> str:
    return name.replace("_", " ")


def encode(names, model_name, pooling="mean", layers=(-1,), template="{}", batch_size=128, device=None):
    """Return an (n, d) float32 matrix, one row per name.

    pooling="cls": hidden state of [CLS]. pooling="mean": mean over the word pieces of the
    *ingredient name only* (template words, [CLS] and [SEP] are left out), which is what
    "averaged BERT" means here. ``layers``: hidden layers to average (-1 = last, 0 = the
    input embeddings).
    """
    allh = encode_all_layers(names, model_name, template, batch_size, device)[pooling]
    return allh[list(layers)].mean(0)


def encode_all_layers(names, model_name, template="{}", batch_size=128, device=None):
    """One forward pass, every layer kept: {"cls": (L+1, n, d), "mean": (L+1, n, d)}."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if "{}" not in template:
        raise ValueError("template must contain {} for the ingredient name")
    prefix, suffix = template.split("{}", 1)
    n_layers = model.config.num_hidden_layers + 1
    shape = (n_layers, len(names), model.config.hidden_size)
    out = {"cls": np.zeros(shape, dtype=np.float32), "mean": np.zeros(shape, dtype=np.float32)}
    with torch.inference_mode():
        for start in range(0, len(names), batch_size):
            batch = [to_text(n) for n in names[start : start + batch_size]]
            texts = [prefix + b + suffix for b in batch]
            enc = tok(texts, padding=True, truncation=True, max_length=64,
                      return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states
            h = torch.stack(hs).cpu()  # (L+1, b, t, d)
            # mean pooling: keep word pieces whose characters fall inside the ingredient name
            lo = torch.tensor([len(prefix)] * len(batch))
            hi = lo + torch.tensor([len(b) for b in batch])
            s, e = offsets[..., 0], offsets[..., 1]
            mask = (e > s) & (s >= lo[:, None]) & (e <= hi[:, None]) & enc["attention_mask"].bool()
            m = mask[None, :, :, None].float()
            sl = slice(start, start + len(batch))
            out["cls"][:, sl] = h[:, :, 0].numpy()
            out["mean"][:, sl] = ((h * m).sum(2) / m.sum(2).clamp(min=1)).numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id, e.g. BAAI/bge-base-en-v1.5")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--layers", nargs="+", type=int, default=[-1], help="hidden layers to average (-1 = last)")
    ap.add_argument("--template", default="{}", help='text around the name, e.g. "a recipe with {}"')
    ap.add_argument("--candidates", help="FlavorGraph nodes_191120.csv")
    ap.add_argument("--data-dir", help="Recipe1MSubs folder (names from vocab_ingrs.pkl)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--online", action="store_true", help="allow downloads from the Hugging Face Hub")
    ap.add_argument("--out", required=True, help="output .npz (words, vectors)")
    args = ap.parse_args(argv)
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    names = ingredient_names(args)
    t0 = time.time()
    vecs = encode(names, args.model, args.pooling, tuple(args.layers), args.template, args.batch_size, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=np.array(names), vectors=vecs,
             meta=np.array(f"{args.model} pooling={args.pooling} layers={args.layers} template={args.template!r}"))
    print(f"{len(names)} ingredients x {vecs.shape[1]} dims -> {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
