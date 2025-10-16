from __future__ import annotations
import io, os, re, random
from urllib.parse import urlparse
from typing import Optional
import requests
from PIL import Image

# Optional deps
try:
    import cv2, numpy as np
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False
    np = None  # type: ignore

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception:
    DEEPFACE_OK = False
    DeepFace = None  # type: ignore

try:
    import imagehash
    IMAGEHASH_OK = True
except Exception:
    IMAGEHASH_OK = False
    imagehash = None  # type: ignore

try:
    from ddgs import DDGS
    DDGS_OK = True
except Exception:
    DDGS_OK = False
    DDGS = None  # type: ignore

# ---- Settings (tweakable at runtime from the UI by mutating these module globals) ----
TIMEOUT = 12
SAFESEARCH = "off"         # 'off' | 'moderate' | 'strict'
MIN_DIM = 200
PREFER_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DISFAVOR_EXTS = {".svg", ".gif", ".ico", ".bmp"}
DDG_MAX_RESULTS = 100
TOP_SAMPLE_K = 20

ENFORCE_WOMAN = True
WOMAN_PROB_THRESHOLD = 0.7
DEEPFACE_ENFORCE_DETECTION = True
DEEPFACE_STRICT_ON_ERROR = True
HOST_DIVERSITY_IN_FALLBACK = True

FACE_MIN_RATIO = 0.3  # display crop (if used by UI)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

class Session:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(DEFAULT_HEADERS)
    def get_bytes(self, url: str, referer: str | None = None, timeout: int = TIMEOUT) -> bytes | None:
        try:
            headers = {}
            if referer: headers["Referer"] = referer
            r = self.s.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:
            return None

SESSION = Session()

def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def candidate_url_from_result(r: dict) -> str | None:
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

def result_min_dim(r: dict) -> int | None:
    w = r.get("width"); h = r.get("height")
    try:
        w = int(w) if w is not None else None
        h = int(h) if h is not None else None
    except Exception:
        return None
    if w and h: return min(w, h)
    return None

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def relevance_score(term: str, extra: str, r: dict) -> float:
    phrase = term.lower()
    title = (r.get("title") or "").lower()
    source = (r.get("source") or "").lower()
    url = (r.get("url") or r.get("image") or r.get("thumbnail") or "").lower()
    text = " ".join([title, source, url])
    score = 0.0
    if phrase in text: score += 2.0
    term_tokens = tokenize(term)
    extra_tokens = tokenize(extra)
    all_tokens = term_tokens | extra_tokens if extra_tokens else term_tokens
    if all_tokens:
        coverage = len([t for t in all_tokens if t in text]) / max(1, len(all_tokens))
        score += 2.5 * coverage
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in PREFER_EXTS: score += 1.0
    if ext in DISFAVOR_EXTS: score -= 1.0
    md = result_min_dim(r)
    if md: score += min(2.0, md / 800.0)
    return score

def ddg_image_results(query: str, max_results: int) -> list[dict]:
    if not DDGS_OK:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            kwargs = {"safesearch": SAFESEARCH, "type_image": "photo", "max_results": max_results}
            for r in ddgs.images(query, **kwargs):
                out.append(r)
    except Exception:
        pass
    return out

def ddg_candidates(term: str, extra: str) -> list[str]:
    q = f"\"{term}\""
    if extra.strip():
        q = f"{q} {extra.strip()}"
    results = ddg_image_results(q, DDG_MAX_RESULTS)
    scored = []
    for r in results:
        url = candidate_url_from_result(r)
        if not url:
            continue
        md = result_min_dim(r)
        if md and md < MIN_DIM:
            continue
        scored.append((relevance_score(term, extra, r), url))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_SAMPLE_K] if TOP_SAMPLE_K and TOP_SAMPLE_K > 0 else scored
    urls = [u for _, u in top]
    random.shuffle(urls)
    return urls

def download_image_bytes(url: str, referer: str | None = None) -> bytes | None:
    return SESSION.get_bytes(url, referer, timeout=TIMEOUT)

# --- Optional analysis ---
def phash_bytes(data: bytes) -> str | None:
    if not IMAGEHASH_OK:
        return None
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return str(imagehash.phash(pil))
    except Exception:
        return None

def looks_like_woman(data: bytes) -> bool:
    if not (ENFORCE_WOMAN and DEEPFACE_OK):
        return True
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(pil) if 'np' in globals() and np is not None else None
        if arr is None:
            import numpy as _np
            arr = _np.array(pil)
        result = DeepFace.analyze(
            img_path=arr,
            actions=['gender'],
            enforce_detection=DEEPFACE_ENFORCE_DETECTION,
            prog_bar=False
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
        return (not DEEPFACE_STRICT_ON_ERROR) == False  # strict -> reject
