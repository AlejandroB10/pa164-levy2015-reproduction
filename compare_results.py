"""Compare a fresh run against the committed reference results.

Usage:
    python compare_results.py results-mine/results.csv

Prints every (configuration, metric) pair whose value differs from results/results.csv by more
than the tolerance. PPMI and SVD are deterministic and must match closely. Multithreaded gensim
SGNS is not deterministic: with a different number of workers its scores can move by up to about
0.03 (most on the small MTurk and WordSim sets), so SGNS gets a looser tolerance.
"""
import argparse
from pathlib import Path

import pandas as pd

REFERENCE = Path(__file__).parent / "results" / "results.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mine", type=Path, help="results.csv from your own run")
    parser.add_argument("--tol", type=float, default=0.01, help="allowed difference for PPMI and SVD (default 0.01)")
    parser.add_argument("--sgns-tol", type=float, default=0.04, help="allowed difference for SGNS (default 0.04)")
    args = parser.parse_args()

    keys = ["config", "metric"]
    ref = pd.read_csv(REFERENCE)[keys + ["method", "value"]]
    mine = pd.read_csv(args.mine)[keys + ["value"]]
    both = ref.merge(mine, on=keys, how="outer", suffixes=("_ref", "_mine"), indicator=True)

    missing = both[both["_merge"] != "both"]
    matched = both[both["_merge"] == "both"].copy()
    matched["diff"] = (matched["value_mine"] - matched["value_ref"]).abs()
    matched["tol"] = matched["method"].map(lambda m: args.sgns_tol if m == "SGNS" else args.tol)
    off = matched[matched["diff"] > matched["tol"]]

    print(f"compared {len(matched)} values (tolerance {args.tol}, SGNS {args.sgns_tol})")
    print(f"largest difference: PPMI/SVD {matched[matched.method != 'SGNS']['diff'].max():.4f}, "
          f"SGNS {matched[matched.method == 'SGNS']['diff'].max():.4f}")
    if len(missing):
        print(f"{len(missing)} rows exist in only one file:")
        print(missing[keys + ["_merge"]].to_string(index=False))
    if len(off):
        print(f"{len(off)} values differ by more than their tolerance:")
        print(off[keys + ["value_ref", "value_mine", "diff"]].to_string(index=False))
    if not len(missing) and not len(off):
        print("OK: your run matches the reference results.")


if __name__ == "__main__":
    main()
