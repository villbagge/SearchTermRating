from __future__ import annotations
import os, re, io
from urllib.parse import urlparse
import requests
from PIL import Image
from .settings import DEFAULT_HEADERS, THUMB_SIZE

# Optional libs guarded
try:
    import imagehash  # type: ignore
    IMAGEHASH_OK = True
except Exception:
    imagehash = None  # type: ignore
    IMAGEHASH_OK = False


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (text or "")).strip("_") or "term"


def normalize_term(name: str) -> str:
    collapsed = re.sub(r"\s+", " ", (name or "").strip()).casefold()
    return slugify(collapsed)


def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


class Session:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(DEFAULT_HEADERS)

    def get_bytes(self, url: str, referer: str | None = None, timeout: int = 10) -> bytes | None:
        try:
            headers = {"Referer": referer} if referer else {}
            r = self.s.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


SESSION = Session()


def phash_bytes(data: bytes) -> str | None:
    if not IMAGEHASH_OK:
        return None
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return str(imagehash.phash(pil))
    except Exception:
        return None

