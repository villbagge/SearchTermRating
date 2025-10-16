from __future__ import annotations
import os, json
from typing import Any, Dict

APP_NAME = "SearchTermRating"

def _config_dir() -> str:
    appdata = os.getenv("APPDATA")
    if appdata:
        path = os.path.join(appdata, APP_NAME)
        os.makedirs(path, exist_ok=True)
        return path
    home = os.path.expanduser("~")
    base = os.path.join(home, ".config", APP_NAME)
    os.makedirs(base, exist_ok=True)
    return base

def _config_path() -> str:
    return os.path.join(_config_dir(), "config.json")

def load_config() -> Dict[str, Any]:
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg: Dict[str, Any]) -> None:
    path = _config_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def get_last_terms_file() -> str | None:
    p = load_config().get("last_terms_file")
    return p if isinstance(p, str) and p.strip() else None

def set_last_terms_file(path: str) -> None:
    if not isinstance(path, str) or not path.strip():
        return
    cfg = load_config()
    cfg["last_terms_file"] = path
    save_config(cfg)
