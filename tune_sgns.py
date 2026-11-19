"""Tune SGNS on Recipe1MSubs val, then report on test once.

    python tune_sgns.py grid [--pilot]   # every config on val -> results/recipes/sgns_val.csv
    python tune_sgns.py test             # gensim defaults vs val-best on test, 3 seeds
                                         # -> results/recipes/sgns_test.csv + embeddings

The grid can be resumed, since configs already in the CSV are skipped. `test` refuses to run a
second time. --force is there to recompute the same numbers, not to try again after changing
the grid because of what test said.
"""
import argparse
import csv
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np  # noqa: E402

from recipes.embeddings import save_embeddings  # noqa: E402
from recipes.evaluate import evaluate_substitutions  # noqa: E402
from recipes.sgns import DEFAULTS, train_sgns, vectors  # noqa: E402
from recipes.vocabulary import load_substitutions, load_vocabulary  # noqa: E402

DATA = Path("data")
CORPUS = DATA / "recipes/ingredients.txt"
RESULTS = Path("results/recipes")
EMBEDDINGS = DATA / "recipes/embeddings"
KNOBS = ["negative", "ns_exponent", "sample", "vector_size", "epochs"]
GRID = dict(negative=[1, 5, 15], ns_exponent=[0.75, 1.0], sample=[0, 1e-5, 1e-4, 1e-3],
            vector_size=[100, 300], epochs=[5, 15, 30, 60, 100])
PILOT = dict(negative=[5], ns_exponent=[0.75], sample=[1e-3], vector_size=[100], epochs=[5, 15, 30])
METRICS = ["mrr", "hit@1", "hit@3", "hit@10", "mrr_covered", "coverage", "n"]
TEST_SEEDS = [1, 2, 3]


def pairs(split):
    vocab = load_vocabulary(DATA / "recipe1msubs/vocab_ingrs.pkl")
    return load_substitutions(DATA / f"recipe1msubs/{split}_comments_subs.pkl", vocab)


def run_config(config, split_pairs, seed=1):
    """Train one model and score it twice, as W and as W+C (W+C needs no extra training)."""
    model = train_sgns(CORPUS, seed=seed, **config)
    rows = []
    for wc in (False, True):
        names, matrix = vectors(model, w_plus_c=wc)
        rows.append({**config, "w+c": wc, "seed": seed,
                     **evaluate_substitutions(names, matrix, split_pairs)})
    return rows, model


def _grid_job(args):
    config, split_pairs = args
    return run_config(config, split_pairs)[0]


def _test_job(args):
    (label, config, wc, seed), test_pairs = args
    model = train_sgns(CORPUS, seed=seed, **config)
    names, matrix = vectors(model, w_plus_c=wc)
    row = {"model": f"sgns-{label}", **config, "w+c": wc, "seed": seed,
           **evaluate_substitutions(names, matrix, test_pairs)}
    return row, (names, matrix)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def config_key(row):
    return tuple(float(row[k]) for k in KNOBS)


def keep_complete_configs(path):
    """Drop what a killed run left half-written, and return the configs that are done.

    A config is done when both its W and its W+C rows are complete. Anything else (a cut last
    line, a config with only one row) is removed from the file so the grid can append to it.
    """
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if None not in r.values() and all(r[k] != "" for k in fields)]
    kinds = {}
    for r in rows:
        kinds.setdefault(config_key(r), set()).add(r["w+c"])
    done = {key for key, seen in kinds.items() if seen == {"False", "True"}}
    keep = [r for r in rows if config_key(r) in done]
    with open(path, newline="") as f:
        unchanged = f.read() == _csv_text(fields, keep)
    if not unchanged:
        with open(path, "w", newline="") as f:
            f.write(_csv_text(fields, keep))
    return done


def _csv_text(fields, rows):
    import io
    out = io.StringIO()
    writer = csv.DictWriter(out, fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def grid(pilot, jobs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("sgns_val_pilot.csv" if pilot else "sgns_val.csv")
    done = keep_complete_configs(out)
    space = PILOT if pilot else GRID
    todo = [dict(zip(KNOBS, values)) for values in itertools.product(*(space[k] for k in KNOBS))]
    todo = [c for c in todo if config_key(c) not in done]
    # start the slowest configs first, otherwise the pool sits waiting on one long job at the end
    todo.sort(key=lambda c: c["epochs"] * c["vector_size"] * (c["negative"] + 1), reverse=True)
    print(f"{len(todo)} configs to train ({len(done)} already in {out})", flush=True)
    val = pairs("val")
    fields = KNOBS + ["w+c", "seed"] + METRICS
    new_file = not out.exists()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(jobs) as pool:
        writer = csv.DictWriter(f, fields)
        if new_file:
            writer.writeheader()
        for i, rows in enumerate(pool.map(_grid_job, [(c, val) for c in todo]), 1):
            writer.writerows(rows)
            f.flush()
            best = max(rows, key=lambda r: r["mrr"])
            config = {k: rows[0][k] for k in KNOBS}
            print(f"[{i}/{len(todo)}] {config} best mrr={best['mrr']:.4f}", flush=True)


def best_val_config():
    rows = read_csv(RESULTS / "sgns_val.csv")
    if not rows:
        sys.exit("no val results: run `python tune_sgns.py grid` first")
    best = max(rows, key=lambda r: float(r["mrr"]))
    config = {k: (int(float(best[k])) if k in ("negative", "vector_size", "epochs") else float(best[k]))
              for k in KNOBS}
    return config, best["w+c"] == "True", best


def test(force):
    out = RESULTS / "sgns_test.csv"
    if out.exists() and not force:
        sys.exit(f"{out} exists: the test split is reported once")
    best, best_wc, best_row = best_val_config()
    print(f"val-best: {best} w+c={best_wc} (val mrr {float(best_row['mrr']):.4f})")
    vanilla = {k: DEFAULTS[k] for k in KNOBS}
    test_pairs = pairs("test")
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    runs = [(label, config, wc, seed) for label, config, wc in
            (("vanilla", vanilla, False), ("tuned", best, best_wc)) for seed in TEST_SEEDS]
    with ProcessPoolExecutor(len(runs)) as pool:
        results = list(pool.map(_test_job, [(r, test_pairs) for r in runs]))
    rows = [row for row, _ in results]
    for (label, _, _, seed), (row, (names, matrix)) in zip(runs, results):
        if seed == TEST_SEEDS[0]:
            save_embeddings(EMBEDDINGS / f"sgns-{label}.txt", names, matrix)
        print(f"{label} seed {seed}: mrr={row['mrr']:.4f}", flush=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, ["model"] + KNOBS + ["w+c", "seed"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    for label in ("vanilla", "tuned"):
        mrr = np.array([r["mrr"] for r in rows if r["model"] == f"sgns-{label}"])
        print(f"sgns-{label}: test MRR {mrr.mean():.4f} ± {mrr.std(ddof=1):.4f} over {len(mrr)} seeds")


def main():
    # gensim seeds each word vector from Python's string hash, which changes per process.
    # Without a fixed seed two runs of the same config give different vectors.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid", help="train every config on val")
    g.add_argument("--pilot", action="store_true", help="epochs axis only, to check timing")
    g.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t = sub.add_parser("test", help="vanilla vs val-best on test, once")
    t.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cmd == "grid":
        grid(args.pilot, args.jobs)
    else:
        test(args.force)


if __name__ == "__main__":
    main()
