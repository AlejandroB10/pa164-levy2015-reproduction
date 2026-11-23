"""Evaluate scorers on Recipe1MSubs substitution pairs.

Two modes:
* ``common`` (default, used for the comparison table): every scorer ranks the *same*
  candidate set (base candidates that every embedding knows) on the *same* pairs (source
  and target both in that set). Numbers are directly comparable.
* ``own``: each scorer ranks its own candidates (base ∩ its vocabulary); pairs it cannot
  score count as misses (reciprocal rank 0), so low coverage is penalized, not hidden.

By default the source ingredient itself is not a candidate (a cosine scorer would always
rank it first; the paper ranks D_ing \\ {s}). ``keep_source=True`` keeps it, as GISMo's code
does. With ``filtered=True`` the other gold targets of the same (recipe, source) are removed
from the ranking too. The paper's text says it does this, but its Table 1 baselines are
reproduced (within 0.1) only *unfiltered*, matching ``filter=False`` in its README, so
unfiltered is the default here.
"""

from __future__ import annotations

from collections import defaultdict

from .data import SubPair
from .metrics import rank_of, summarize


def _gold_targets(pairs):
    gold = defaultdict(set)
    for p in pairs:
        gold[(p.recipe_id, p.source)].add(p.target)
    return gold


def rank_pairs(scorer, pairs: list[SubPair], candidates: list[str], filtered=False, keep_source=False):
    """Rank of each pair's target, or 0 when the scorer cannot rank that pair."""
    cand_set = set(candidates)
    gold = _gold_targets(pairs) if filtered else None
    by_source = defaultdict(list)
    for i, p in enumerate(pairs):
        by_source[p.source].append(i)

    ranks = [0] * len(pairs)
    for src, idxs in by_source.items():
        if src not in cand_set:
            continue
        cands = list(candidates) if keep_source else [c for c in candidates if c != src]
        pos = {c: j for j, c in enumerate(cands)}
        s = scorer.scores(src, cands)
        for i in idxs:
            p = pairs[i]
            if p.target not in pos:
                continue
            excl = [pos[t] for t in gold[(p.recipe_id, src)] if t != p.target and t in pos] if filtered else ()
            ranks[i] = rank_of(s, pos[p.target], excl)
    return ranks


def candidates_for(scorer, base: list[str]) -> list[str]:
    return list(base) if scorer.vocab is None else [c for c in base if c in scorer.vocab]


def evaluate(scorers, pairs, base_candidates, mode="common", filtered=False, keep_source=False):
    """Return (rows, per_pair) where rows = one dict of metrics per scorer."""
    base = sorted(set(base_candidates))
    if mode == "common":
        cands = base
        for sc in scorers:
            cands = candidates_for(sc, cands)
        keep = set(cands)
        eval_pairs = [p for p in pairs if p.source in keep and p.target in keep and p.source != p.target]
    elif mode == "own":
        eval_pairs = [p for p in pairs if p.source != p.target]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    rows, per_pair = [], {}
    for sc in scorers:
        own = cands if mode == "common" else candidates_for(sc, base)
        ranks = rank_pairs(sc, eval_pairs, own, filtered, keep_source)
        covered = sum(1 for r in ranks if r)
        row = {"model": sc.name, "mode": mode, "filtered": filtered, "keep_source": keep_source,
               **summarize(ranks)}
        row.update(
            n_candidates=len(own),
            covered=covered,
            n_split=len(pairs),
            coverage=100.0 * covered / len(pairs) if pairs else float("nan"),
        )
        rows.append(row)
        per_pair[sc.name] = ranks
    return rows, eval_pairs, per_pair
