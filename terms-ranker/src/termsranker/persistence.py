from __future__ import annotations
import csv, os, re
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from .core import Term, DEFAULT_RATING, SIGMA0

def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "term"

def normalize_term(name: str) -> str:
    collapsed = re.sub(r"\s+", " ", (name or "").strip()).casefold()
    return slugify(collapsed)

def _parse_row(row: list[str]) -> Term | None:
    if not row:
        return None
    name = (row[0] or "").strip()
    if not name:
        return None
    rating = DEFAULT_RATING
    games = 0
    sigma = SIGMA0
    if len(row) >= 2:
        try: rating = float(row[1])
        except: pass
    if len(row) >= 3:
        try: games = int(float(row[2]))
        except: pass
    if len(row) >= 4:
        try: sigma = float(row[3])
        except: pass
    return Term(name=name, rating=rating, games=games, sigma=sigma)

def load_terms(path: str) -> list[Term]:
    out: list[Term] = []
    seen_norm: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                t = _parse_row(row)
                if t is None:
                    continue
                norm = normalize_term(t.name)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                out.append(t)
    except Exception:
        return out
    return out

def save_terms(path: str, terms: list[Term]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Always write 4 columns: name, rating, games, sigma
        for t in terms:
            w.writerow([t.name, f"{t.rating:.6f}", str(int(t.games)), f"{t.sigma:.6f}"])
    os.replace(tmp, path)

def used_cache_path(terms_path: str) -> str:
    base = os.path.splitext(os.path.abspath(terms_path))[0]
    return base + "_used_urls.txt"

def seen_hashes_path(terms_path: str) -> str:
    base = os.path.splitext(os.path.abspath(terms_path))[0]
    return base + "_seen_hashes.txt"

def load_used(path: str) -> dict[str, set[str]]:
    used: dict[str, set[str]] = defaultdict(set)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                term, url = line.rstrip("\n").split(" ", 1)
                used[term].add(url)
    except Exception:
        pass
    return used

def save_used(path: str, used: dict[str, set[str]]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for term, urls in used.items():
            for u in sorted(urls):
                f.write(f"{term} {u}\n")
    os.replace(tmp, path)

def load_seen_hashes(path: str) -> set[str]:
    hashes: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h: hashes.add(h)
    except Exception:
        pass
    return hashes

def save_seen_hashes(path: str, hashes: set[str]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for h in sorted(hashes):
            f.write(h + "\n")
    os.replace(tmp, path)
