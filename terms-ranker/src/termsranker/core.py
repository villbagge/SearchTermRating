from __future__ import annotations
from dataclasses import dataclass
import math
import random
import re
from typing import Optional, Dict

# ---- Elo + uncertainty constants ----
DEFAULT_RATING = 1500.0
SIGMA0 = 350.0
SIGMA_FLOOR = 60.0
BASE_K = 36.0
SIMILARITY_DAMP_MAX = 0.5  # up to 50% damp when terms are nearly identical

# ---- Weighted sampling knobs ----
UNRANKED_BOOST = 0.20
HIGH_SPREAD_BOOST = 0.10
HIGH_SPREAD_THRESHOLD = 0.60

# ---- Recency cooldown knobs (can be updated at runtime via Settings) ----
COOLDOWN_WINDOW = 6           # how many recent rounds are considered "too recent"
COOLDOWN_MIN_FACTOR = 0.35    # minimal multiplier for just-shown items

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

def recency_factor(name: str, recency_map: Optional[Dict[str, int]]) -> float:
    """Return a multiplier in (COOLDOWN_MIN_FACTOR..1] based on rounds since last shown.
    0 means shown this round (shouldn't happen for candidates), 1 means unseen within window.
    """
    if not recency_map:
        return 1.0
    rounds_since = recency_map.get(name)
    if rounds_since is None:
        return 1.0
    if rounds_since >= COOLDOWN_WINDOW:
        return 1.0
    # Linear ramp from COOLDOWN_MIN_FACTOR up to 1 across the window
    # rounds_since = 0 -> min_factor ; rounds_since = COOLDOWN_WINDOW -> 1.0
    frac = max(0.0, min(1.0, rounds_since / max(1.0, COOLDOWN_WINDOW)))
    return COOLDOWN_MIN_FACTOR + (1.0 - COOLDOWN_MIN_FACTOR) * frac

def term_weight(t: Term, recency_map: Optional[Dict[str, int]] = None) -> float:
    # Base exploration weight
    w = 1.0
    if getattr(t, "games", 0) == 0:
        w *= (1.0 + UNRANKED_BOOST)
    if normalized_sigma(t) >= HIGH_SPREAD_THRESHOLD:
        w *= (1.0 + HIGH_SPREAD_BOOST)
    # Add a gentle experience downweight: the more games, the smaller the weight
    # This doesn't zero-out frequent terms; it just nudges selection toward less-explored ones.
    w *= 1.0 / (1.0 + 0.12 * max(0, t.games))  # alpha = 0.12 (tunable)
    # Apply recency cooldown
    w *= recency_factor(t.name, recency_map)
    return max(1e-9, float(w))

def weighted_sample_terms(terms: list[Term], k: int, recency_map: Optional[Dict[str, int]] = None, exclude: set[str] | None = None) -> list[Term]:
    if not terms:
        return []
    if exclude:
        pool = [t for t in terms if t.name not in exclude]
    else:
        pool = terms
    if not pool:
        return []
    k = min(k, len(pool))
    keys = []
    for t in pool:
        w = term_weight(t, recency_map)
        r = random.random() or 1e-12
        key = r ** (1.0 / w)  # larger key -> more likely selected
        keys.append((key, t))
    keys.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in keys[:k]]
