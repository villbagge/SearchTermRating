from __future__ import annotations
import os, re, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .persistence import slugify
from .images import phash_bytes

# ----- Tuning -----
# Hamming distance threshold for perceptual duplicates.
# 0 = identical phash; 1-3 = very similar; try 2 (your current setting)
THRESHOLD = 2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FNAME_RE = re.compile(
    r"^(?P<slug>.+?)_(?P<rating>-?\d+?)_(?P<ts>\d{8}-\d{6})\.(?P<ext>jpg|jpeg|png|webp|bmp|tif|tiff)$",
    re.IGNORECASE
)

@dataclass
class ImgMeta:
    path: Path
    size: int
    mtime: float
    sha1: str
    phash: str | None
    rating: int | None

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _load_meta(fp: Path) -> ImgMeta | None:
    try:
        data = fp.read_bytes()
    except Exception:
        return None
    try:
        phex = phash_bytes(data)  # may return None
    except Exception:
        phex = None
    try:
        st = fp.stat()
        size = st.st_size
        mtime = st.st_mtime
    except Exception:
        size = 0
        mtime = 0.0
    sha1 = _sha1_bytes(data)
    # try parse rating from name
    rating = None
    m = FNAME_RE.match(fp.name)
    if m:
        try:
            rating = int(m.group("rating"))
        except Exception:
            rating = None
    return ImgMeta(path=fp, size=size, mtime=mtime, sha1=sha1, phash=phex, rating=rating)

def _ham(a_hex: str, b_hex: str) -> int:
    try:
        return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")
    except Exception:
        return 9999

def _pick_best(imgs: List[ImgMeta]) -> ImgMeta:
    # Prefer highest rating; if missing tie, prefer newest mtime; then shortest path (stable)
    imgs_sorted = sorted(
        imgs,
        key=lambda m: (
            -(m.rating if m.rating is not None else -10**9),  # None rating => lowest priority
            -m.mtime,
            len(str(m.path))
        )
    )
    return imgs_sorted[0]

def _cluster_within_threshold(imgs: List[ImgMeta], threshold: int) -> List[List[ImgMeta]]:
    """Greedy clustering: exact SHA1 groups first, then pHash groups with threshold."""
    remaining = list(imgs)
    clusters: List[List[ImgMeta]] = []

    # Step 1: exact-duplicate clusters by SHA1
    by_sha: Dict[str, List[ImgMeta]] = {}
    for m in remaining:
        by_sha.setdefault(m.sha1, []).append(m)
    for sha, group in list(by_sha.items()):
        if len(group) > 1:
            clusters.append(group)
    # Remove those from remaining so we don't re-cluster them
    clustered_paths = {m.path for group in clusters for m in group}
    remaining = [m for m in remaining if m.path not in clustered_paths]

    # Step 2: perceptual clusters (greedy)
    used = set()
    for i, m in enumerate(remaining):
        if m.path in used:
            continue
        group = [m]
        used.add(m.path)
        for j in range(i+1, len(remaining)):
            n = remaining[j]
            if n.path in used:
                continue
            if not m.phash or not n.phash:
                continue  # can't compare
            if _ham(m.phash, n.phash) <= threshold:
                group.append(n)
                used.add(n.path)
        if len(group) > 1:
            clusters.append(group)

    return clusters

def _collect_term_dirs(base_dir: Path) -> List[Path]:
    ranked_dir = (base_dir / "ranked_images")
    if not ranked_dir.is_dir():
        return []
    return [p for p in ranked_dir.iterdir() if p.is_dir()]

def _list_images(term_dir: Path) -> List[Path]:
    return [p for p in term_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def dedupe_ranked_images(base_dir: Path, threshold: int = THRESHOLD) -> dict:
    """
    Walk ranked_images/<term> folders.
    Find exact and perceptual duplicates, keep the best, delete the rest.
    Returns a summary dict with totals.
    """
    base_dir = Path(base_dir)
    term_dirs = _collect_term_dirs(base_dir)

    totals = {"scanned": 0, "kept": 0, "deleted": 0, "clusters": 0}
    per_term: Dict[str, Dict[str, int]] = {}

    for term_dir in term_dirs:
        files = _list_images(term_dir)
        metas: List[ImgMeta] = []
        for f in files:
            m = _load_meta(f)
            if m:
                metas.append(m)
        if not metas:
            continue

        totals["scanned"] += len(metas)
        per_term_stats = {"scanned": len(metas), "kept": 0, "deleted": 0, "clusters": 0}

        clusters = _cluster_within_threshold(metas, threshold=threshold)
        if not clusters:
            per_term_stats["kept"] += len(metas)
            totals["kept"] += len(metas)
            per_term[term_dir.name] = per_term_stats
            continue

        # Mark files to delete in each cluster
        to_delete: List[Path] = []
        for group in clusters:
            per_term_stats["clusters"] += 1
            best = _pick_best(group)
            # delete all except the best
            for m in group:
                if m.path != best.path:
                    to_delete.append(m.path)

        # Apply deletions
        deleted_count = 0
        for fp in to_delete:
            try:
                os.remove(fp)
                deleted_count += 1
            except Exception:
                pass

        kept_count = len(metas) - deleted_count
        per_term_stats["deleted"] += deleted_count
        per_term_stats["kept"] += kept_count
        totals["deleted"] += deleted_count
        totals["kept"] += kept_count
        totals["clusters"] += per_term_stats["clusters"]
        per_term[term_dir.name] = per_term_stats

    return {"totals": totals, "per_term": per_term, "threshold": threshold}
