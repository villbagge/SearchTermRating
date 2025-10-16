from __future__ import annotations
from typing import Optional
import io
import requests
from PIL import Image
# from ddgs import DDGS  # hook this up when wiring image search

def fetch_image_bytes(url: str, timeout: int = 12) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
