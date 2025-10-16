from __future__ import annotations
import math
from dataclasses import dataclass
from .settings import (
    DEFAULT_RATING, SIGMA0, SIGMA_FLOOR, BASE_K, SIMILARITY_DAMP_MAX,
    UNRANKED_BOOST, HIGH_SPREAD_BOOST, HIGH_SPREAD_THRESHOLD,
)
from .utils import tokenize


@dataclass
class Term:
    name: str
    rating: float = DEFAULT_RATING
    games: int = 0
    sigma: float = SIGMA0


def jaccard_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def elo_update(winner: Term, losers: list[Term]):
    for loser in losers:
        diff = loser.rating - winner.rating
        expected = 1.0 / (1.0 + math.pow(10.0, diff / 400.0))
        u_scale = min(2.0, max(0.5, (winner.sigma + loser.sigma) / (2 * SIGMA0)))
        damp = 1.0 - SIMILARITY_DAMP_MAX * jaccard_similarity(winner.name, loser.name)
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
    if int(getattr(t, "games", 0)) == 0:
        w *= (1.0 + UNRANKED_BOOST)
    if normalized_sigma(t) >= HIGH_SPREAD_THRESHOLD:
        w *= (1.0 + HIGH_SPREAD_BOOST)
    return max(1e-9, float(w))
