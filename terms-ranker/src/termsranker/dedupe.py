from __future__ import annotations
import os, io, shutil, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from .persistence import slugify
from .images import phash_bytes

# Fixed parameters
THRESHOLD = 4  # Hamming distance threshold for similarity (lower = stricter)

@dataclass
class ImgInfo:
    path: Path
    phash: int
    size: Tuple[int, int]  # (w,h)
    bytes: int
    mtime: float

def _read_phash(path: Path) -> ImgInfo | None:
    try:
        data = path.read_bytes()
        h = phash_bytes(data)
        if not h:
            return None
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            w, hpx = im.size
        st = path.stat()
        return ImgInfo(
            path=path,
            phash=int(h, 16) if isinstance(h, str) else int(h),
            size=(w, hpx),
            bytes=st.st_size,
            mtime=st.st_mtime,
        )
    except Exception:
        return None

def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _choose_keeper(files: List[ImgInfo]) -> ImgInfo:
    def score(info: ImgInfo):
        w, h = info.size
        return (w*h, info.bytes, info.mtime)
    return max(files, key=score)

def _cluster_by_phash(infos: List[ImgInfo], threshold: int) -> List[List[ImgInfo]]:
    clusters: List[List[ImgInfo]] = []
    reps: List[int] = []
    for info in infos:
        placed = False
        for i, rep in enumerate(reps):
            if _hamming(info.phash, rep) <= threshold:
                clusters[i].append(info)
                placed = True
                break
        if not placed:
            clusters.append([info])
            reps.append(info.phash)
    return clusters

def _iter_term_dirs(ranked_dir: Path) -> List[Path]:
    if not ranked_dir.exists():
        return []
    return sorted([p for p in ranked_dir.iterdir() if p.is_dir()])

def dedupe_ranked_images(base_dir: Path) -> Dict[str, Dict[str, int]]:
    ranked_dir = base_dir / "ranked_images"
    summary: Dict[str, Dict[str, int]] = {"terms": {}, "totals": {"kept": 0, "deleted": 0, "scanned": 0}}

    for term_dir in _iter_term_dirs(ranked_dir):
        files = sorted(term_dir.glob("*.jpg"))
        if not files:
            continue
        infos: List[ImgInfo] = []
        for p in files:
            info = _read_phash(p)
            if info:
                infos.append(info)

        summary["totals"]["scanned"] += len(infos)
        if not infos:
            continue

        clusters = _cluster_by_phash(infos, THRESHOLD)
        kept = deleted = 0

        for cluster in clusters:
            if len(cluster) <= 1:
                kept += 1
                continue
            keep = _choose_keeper(cluster)
            kept += 1
            for dup in cluster:
                if dup.path == keep.path:
                    continue
                try:
                    dup.path.unlink(missing_ok=True)
                    deleted += 1
                except Exception:
                    pass

        summary["terms"][term_dir.name] = {
            "kept": kept,
            "deleted": deleted,
            "clusters": len(clusters)
        }
        summary["totals"]["kept"] += kept
        summary["totals"]["deleted"] += deleted

    return summary

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Remove visually redundant images (auto-delete).")
    ap.add_argument("--terms", required=True, help="Path to your terms file (to locate ranked_images next to it).")
    args = ap.parse_args()

    terms_path = Path(args.terms).expanduser().resolve()
    base_dir = terms_path.parent

    summary = dedupe_ranked_images(base_dir)
    print("Dedup summary (threshold=10, duplicates deleted):")
    for term, stats in sorted(summary["terms"].items()):
        print(f"  {term:30s} kept={stats['kept']:3d} deleted={stats['deleted']:3d} clusters={stats['clusters']:3d}")
    t = summary["totals"]
    print(f"\nTotals: scanned={t['scanned']} kept={t['kept']} deleted={t['deleted']}")

if __name__ == "__main__":
    main()
