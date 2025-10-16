from __future__ import annotations
import os, random, io
from urllib.parse import urlparse
from PIL import Image, ImageOps
from .settings import (
    DDG_MAX_RESULTS, SAFESEARCH, MIN_DIM, PREFER_EXTS, DISFAVOR_EXTS, TOP_SAMPLE_K, THUMB_SIZE,
    TIMEOUT, ENFORCE_WOMAN, WOMAN_PROB_THRESHOLD, DEEPFACE_ENFORCE_DETECTION, DEEPFACE_STRICT_ON_ERROR,
)
from .utils import SESSION, tokenize

# Optional imports guarded
try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover
    DDGS = None  # type: ignore

try:
    import imagehash  # noqa: F401
    IMAGEHASH_OK = True
except Exception:
    IMAGEHASH_OK = False

try:
    import numpy as np  # type: ignore
    import cv2  # type: ignore
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False
    np = None  # type: ignore
    cv2 = None  # type: ignore

try:
    from deepface import DeepFace  # type: ignore
    DEEPFACE_OK = True
except Exception:
    DeepFace = None  # type: ignore
    DEEPFACE_OK = False


def ddg_image_results(query: str, max_results: int) -> list[dict]:
    out: list[dict] = []
    if DDGS is None:
        return out
    try:
        with DDGS() as ddgs:
            kwargs = {"safesearch": SAFESEARCH, "type_image": "photo", "max_results": max_results}
            for r in ddgs.images(query, **kwargs):
                out.append(r)
    except Exception:
        pass
    return out


def _candidate_url_from_result(r: dict) -> str | None:
    for key in ("image", "url", "thumbnail"):
        u = r.get(key)
        if isinstance(u, str) and u.startswith("http"):
            ext = os.path.splitext(urlparse(u).path)[1].lower()
            if ext and PREFER_EXTS and ext in PREFER_EXTS:
                return u
            if ext and DISFAVOR_EXTS and ext in DISFAVOR_EXTS:
                continue
            return u
    return None


def _result_min_dim(r: dict) -> int | None:
    w = r.get("width"); h = r.get("height")
    try:
        w = int(w) if w is not None else None
        h = int(h) if h is not None else None
    except Exception:
        return None
    if w and h:
        return min(w, h)
    return None


def _relevance_score(term: str, extra: str, r: dict) -> float:
    phrase = term.lower()
    title = (r.get("title") or "").lower()
    source = (r.get("source") or "").lower()
    url = (r.get("url") or r.get("image") or r.get("thumbnail") or "").lower()
    text = " ".join([title, source, url])
    score = 0.0
    if phrase in text:
        score += 2.0
    term_tokens = tokenize(term)
    extra_tokens = tokenize(extra)
    all_tokens = term_tokens | extra_tokens if extra_tokens else term_tokens
    if all_tokens:
        coverage = len([t for t in all_tokens if t in text]) / max(1, len(all_tokens))
        score += 2.5 * coverage
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in PREFER_EXTS: score += 1.0
    if ext in DISFAVOR_EXTS: score -= 1.0
    md = _result_min_dim(r)
    if md: score += min(2.0, md / 800.0)
    return score


def ddg_candidates(term: str, extra: str) -> list[str]:
    q = f'"{term}"'
    if extra.strip():
        q = f"{q} {extra.strip()}"
    results = ddg_image_results(q, DDG_MAX_RESULTS)
    scored: list[tuple[float, str]] = []
    for r in results:
        url = _candidate_url_from_result(r)
        if not url:
            continue
        md = _result_min_dim(r)
        if md and md < MIN_DIM:
            continue
        scored.append((_relevance_score(term, extra, r), url))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_SAMPLE_K] if TOP_SAMPLE_K and TOP_SAMPLE_K > 0 else scored
    urls = [u for _, u in top]
    random.shuffle(urls)
    return urls


def download_image_bytes(url: str, referer: str | None = None) -> bytes | None:
    return SESSION.get_bytes(url, referer, timeout=TIMEOUT)


def looks_like_woman(data: bytes) -> bool:
    if not (ENFORCE_WOMAN and DEEPFACE_OK):
        return True
    try:
        from PIL import Image  # local import
        import numpy as _np  # type: ignore
        pil = Image.open(io.BytesIO(data)).convert("RGB")  # type: ignore[name-defined]
        arr = _np.array(pil)
        result = DeepFace.analyze(
            img_path=arr,
            actions=['gender'],
            enforce_detection=DEEPFACE_ENFORCE_DETECTION,
            prog_bar=False,
        )
        if isinstance(result, list) and result:
            result = result[0]
        dom = str(result.get('dominant_gender', '')).lower()
        gender_dict = result.get('gender') or {}
        prob_woman = 0.0
        for k, v in gender_dict.items():
            if str(k).lower().startswith("woman"):
                try:
                    prob_woman = float(v) / 100.0 if float(v) > 1.0 else float(v)
                except Exception:
                    pass
                break
        if dom.startswith('woman') and prob_woman == 0.0:
            return True
        return prob_woman >= WOMAN_PROB_THRESHOLD
    except Exception:
        return (not DEEPFACE_STRICT_ON_ERROR) == False  # True -> reject when strict
