from __future__ import annotations
import csv, os
from .core import Term, DEFAULT_RATING

def load_terms(path: str) -> list[Term]:
    terms: list[Term] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row: continue
            name = (row[0] or "").strip()
            if not name: continue
            key = " ".join(name.split()).casefold()
            if key in seen: continue
            seen.add(key)
            rating = DEFAULT_RATING
            if len(row) >= 2:
                try: rating = float(row[1])
                except: pass
            terms.append(Term(name=name, rating=rating))
    return terms

def save_terms(path: str, terms: list[Term]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for t in terms:
            w.writerow([t.name, f"{t.rating:.1f}"])
    os.replace(tmp, path)
