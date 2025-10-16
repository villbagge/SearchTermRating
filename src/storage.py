from __future__ import annotations
import os, csv
from collections import defaultdict
from .model import Term
from .utils import normalize_term


def load_terms(path: str) -> list[Term]:
    out: list[Term] = []
    seen_norm: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                name = (row[0] or "").strip()
                if not name:
                    continue
                norm = normalize_term(name)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                rating = None
                if len(row) >= 2:
                    try:
                        rating = float(row[1])
                    except Exception:
                        rating = None
                out.append(Term(name, rating=rating if rating is not None else Term().rating))
    except Exception:
        return out
    return out


def save_terms(path: str, terms: list[Term]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for t in terms:
            w.writerow([t.name, f"{t.rating:.1f}"])
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
                if h:
                    hashes.add(h)
    except Exception:
        pass
    return hashes


def save_seen_hashes(path: str, hashes: set[str]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for h in sorted(hashes):
            f.write(h + "\n")
    os.replace(tmp, path)
