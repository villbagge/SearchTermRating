from dataclasses import dataclass

DEFAULT_RATING = 1500.0
SIGMA0 = 350.0
SIGMA_FLOOR = 60.0
BASE_K = 36.0

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
        u_scale = min(2.0, max(0.5, (winner.sigma + loser.sigma) / (2 * SIGMA0)))
        k = BASE_K * u_scale
        delta = k * (1.0 - expected)
        winner.rating += delta
        loser.rating -= delta
        winner.games += 1
        loser.games += 1
        winner.sigma = max(SIGMA_FLOOR, winner.sigma * 0.98)
        loser.sigma  = max(SIGMA_FLOOR, loser.sigma * 0.98)
