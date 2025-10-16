from __future__ import annotations

# ---- UI / layout ----
TIMEOUT = 12
PADDING = 12
THUMB_SIZE = (520, 520)
SELF_TEST_EACH_BATCH = False

# ---- Elo defaults ----
DEFAULT_RATING = 1500.0
SIGMA0 = 350.0
SIGMA_FLOOR = 60.0
BASE_K = 36.0
SIMILARITY_DAMP_MAX = 0.5

# ---- Search / filters ----
DDG_MAX_RESULTS = 100
SAFESEARCH = "off"  # 'off'|'moderate'|'strict'
MIN_DIM = 200
PREFER_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DISFAVOR_EXTS = {".svg", ".gif", ".ico", ".bmp"}
TOP_SAMPLE_K = 20

# ---- Gender filtering (optional; code tolerates absence of deps) ----
ENFORCE_WOMAN = True
WOMAN_PROB_THRESHOLD = 0.7
DEEPFACE_ENFORCE_DETECTION = True
DEEPFACE_STRICT_ON_ERROR = True

# ---- Misc runtime flags ----
HOST_DIVERSITY_IN_FALLBACK = True
FACE_MIN_RATIO = 0.3

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

UNRANKED_BOOST = 0.20
HIGH_SPREAD_BOOST = 0.10
HIGH_SPREAD_THRESHOLD = 0.60
