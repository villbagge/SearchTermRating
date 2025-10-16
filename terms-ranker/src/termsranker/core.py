from __future__ import annotations
from dataclasses import dataclass
import math
import random
import re

# ---- Elo + uncertainty + similarity-damp constants ----
DEFAULT_RATING = 1500.0
SIGMA0 = 350.0
SIGMA_FLOOR = 60.0
BASE_K = 36.0
SIMILARITY_DAMP_MAX = 0.5  # up to 50% damp when terms are nearly identical

# ---- Weighted sampling knobs ----
UNRANKED_BOOST = 0.20
HIGH_SPREAD_BOOST = 0.10
HIGH_SPREAD_THRESHOLD = 0.60

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))

@dataclass
class Term:
    name: str
    rating: float = DEFAULT_RATING
    games: int = 0
    sigma: float = SIGMA0

def elo_update(winner: Term, losers: list[Term]) -> None:
    for loser in losers:
        diff = loser.rating - winner.rating
        expected = 1.0 / (1.0 + 10.0 ** (diff / 400.0))
        # uncertainty scale (cap 0.5x..2x)
        u_scale = min(2.0, max(0.5, (winner.sigma + loser.sigma) / (2 * SIGMA0)))
        # similarity damp (avoid huge changes for near-identical terms)
        sim = jaccard_similarity(winner.name, loser.name)
        damp = 1.0 - SIMILARITY_DAMP_MAX * sim
        k = BASE_K * u_scale * damp
        delta = k * (1.0 - expected)
        winner.rating += delta
        loser.rating -= delta
        winner.games += 1
        loser.games += 1
        winner.sigma = max(SIGMA_FLOOR, winner.sigma * 0.98)
        loser.sigma  = max(SIGMA_FLOOR, loser.sigma * 0.98)

def normalized_sigma(t: Term) -> float:
    denom = max(1e-9, (SIGMA0 - SIGMA_FLOOR))
    return max(0.0, min(1.0, (t.sigma - SIGMA_FLOOR) / denom))

def term_weight(t: Term) -> float:
    w = 1.0
    if getattr(t, "games", 0) == 0:
        w *= (1.0 + UNRANKED_BOOST)
    if normalized_sigma(t) >= HIGH_SPREAD_THRESHOLD:
        w *= (1.0 + HIGH_SPREAD_BOOST)
    return max(1e-9, float(w))

def weighted_sample_terms(terms: list[Term], k: int) -> list[Term]:
    if not terms:
        return []
    k = min(k, len(terms))
    keys = []
    for t in terms:
        w = term_weight(t)
        r = random.random() or 1e-12
        key = r ** (1.0 / w)  # larger key -> more likely selected
        keys.append((key, t))
    keys.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in keys[:k]]
