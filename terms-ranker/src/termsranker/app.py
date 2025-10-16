from __future__ import annotations
import io, os, sys, threading, json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps

from .core import Term, elo_update, weighted_sample_terms, COOLDOWN_WINDOW
from .persistence import (
    load_terms, save_terms, used_cache_path, seen_hashes_path, load_used, save_used,
    load_seen_hashes, save_seen_hashes, slugify
)
from .images import (
    ddg_candidates, download_image_bytes, looks_like_woman, phash_bytes, host_of
)

# -------------------- Config --------------------
CONFIG_PATH = os.path.join(Path.home(), ".termsranker.json")
PADDING = 6
GRID_GAP = 2
DEBUG_OVERLAY = True  # set False to hide host text on tiles

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2)
    except Exception: pass

# -------------------- Canvas helpers (main thread only) --------------------
def draw_title(canvas: "tk.Canvas", text: str):
    # persistent label at top-left
    try:
        canvas.delete("title")
        canvas.create_text(8, 8, text=text, anchor="nw", fill="#ddd", tags="title")
    except Exception:
        pass

def draw_debug(canvas: "tk.Canvas", text: str):
    if not DEBUG_OVERLAY: return
    try:
        w = max(1, int(canvas.winfo_width()))
    except Exception:
        w = 400
    try:
        canvas.delete("dbg")
        canvas.create_text(w-8, 8, text=text, anchor="ne", fill="#aaa", font=("Segoe UI", 9), tags="dbg")
    except Exception:
        pass

def canvas_show_error(canvas: "tk.Canvas", msg: str):
    try:
        canvas.delete("img"); canvas.delete("err")
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
        canvas.create_text(w//2, h//2, text=msg, fill="#ddd", tags="err")
    except Exception:
        pass

def render_to_canvas(canvas: "tk.Canvas", pil_img: Image.Image):
    # Render image but DO NOT touch 'title' or 'dbg' tags so labels persist.
    try:
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
    except Exception:
        w = h = 0
    if w < 10 or h < 10: w, h = 400, 400
    try: resample = Image.Resampling.LANCZOS
    except Exception: resample = Image.ANTIALIAS  # type: ignore
    fitted = ImageOps.contain(pil_img.convert("RGB"), (w, h), method=resample)
    tkimg = ImageTk.PhotoImage(fitted)
    canvas.delete("img")
    canvas.create_image(w//2, h//2, image=tkimg, anchor="center", tags="img")
    canvas.image = tkimg

# -------------------- Prefetch types (atomic / coherent) --------------------
@dataclass(frozen=True)
class PrefetchEntry:
    term: Term
    url: str | None
    data: bytes | None

@dataclass(frozen=True)
class PrefetchBatch:
    entries: list[PrefetchEntry]  # length 4

# -------------------- App --------------------
class App:
    def __init__(self, root: tk.Tk, terms_path: str | None):
        self.root = root
        self.terms_path = terms_path or load_config().get("last_terms_path")
        if not self.terms_path:
            self.terms_path = filedialog.askopenfilename(
                title="Select terms file",
                filetypes=[("Text/CSV","*.txt *.csv"),("All files","*.*")]
            )
            if not self.terms_path: print("No file selected."); sys.exit(0)
        cfg = load_config(); cfg["last_terms_path"] = self.terms_path; save_config(cfg)

        self.terms: list[Term] = load_terms(self.terms_path)
        if not self.terms:
            messagebox.showerror("No Terms","The file is empty. Add terms (one per line) or 'term,rating[,games,sigma]'.")
            sys.exit(1)

        # caches
        self.used_path = used_cache_path(self.terms_path); self.used = load_used(self.used_path)
        self.seen_hashes_path = seen_hashes_path(self.terms_path); self.seen_hashes = load_seen_hashes(self.seen_hashes_path)

        # selection recency
        self.round_id = 0
        self.recency_map: dict[str,int] = {}

        # UI state
        self.extra_var = tk.StringVar(value="")
        self.frames: list[ttk.Frame] = []
        self.labels: list[tk.Canvas] = []
        self.overlay_btns: list[ttk.Button] = []
        self.current_pils: list[Image.Image | None] = [None]*4
        self.current_four: list[Term] = []
        self.current_urls: list[str | None] = [None]*4
        self.current_raw_bytes: list[bytes | None] = [None]*4

        # prefetch queue
        self.prefetch_lock = threading.Lock()
        self.prefetch_queue: list[PrefetchBatch] = []

        # ---- UI ----
        self.root.title("Terms Ranker")
        root.grid_rowconfigure(1, weight=1); root.grid_columnconfigure(0, weight=1)

        top = ttk.Frame(root, padding=PADDING); top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        leftbar = ttk.Frame(top); leftbar.grid(row=0, column=0, sticky="w")
        ttk.Label(leftbar, text="Click the best image").pack(side=tk.LEFT)
        ttk.Label(leftbar, text="  |  Extra terms:").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(leftbar, textvariable=self.extra_var, width=30).pack(side=tk.LEFT, padx=(0,10))
        rightbar = ttk.Frame(top); rightbar.grid(row=0, column=1, sticky="e")
        for text, cmd in [("Gallery", self._open_gallery), ("Settings…", self._open_settings),
                          ("Open Terms File", self._open_file), ("Remove Terms…", self._open_remove_terms),
                          ("Add Terms…", self._open_add_terms)]:
            ttk.Button(rightbar, text=text, command=cmd).pack(side=tk.LEFT, padx=(0,6))

        grid = ttk.Frame(root, padding=0); grid.grid(row=1, column=0, sticky="nsew")
        for r in (0,1): grid.grid_rowconfigure(r, weight=1)
        for c in (0,1): grid.grid_columnconfigure(c, weight=1)

        for i in range(4):
            r, c = divmod(i, 2)
            slot = ttk.Frame(grid, padding=0); slot.grid(row=r, column=c, sticky="nsew", padx=GRID_GAP, pady=GRID_GAP)
            slot.grid_rowconfigure(0, weight=1); slot.grid_columnconfigure(0, weight=1)
            cvs = tk.Canvas(slot, highlightthickness=0, bg="#222"); cvs.grid(row=0, column=0, sticky="nsew")
            cvs.bind("<Configure>", lambda e, idx=i: self._on_canvas_resize(idx))
            cvs.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))
            btn = ttk.Button(cvs, text="OTHER", command=lambda idx=i: self._on_other(idx))
            def place_btn(canvas=cvs, b=btn):
                try:
                    w = max(1, canvas.winfo_width()); h = max(1, canvas.winfo_height())
                    b.place(x=w-8, y=h-8, anchor="se")
                except Exception: pass
            cvs.bind("<Configure>", lambda e, pb=place_btn: pb()); place_btn()
            self.frames.append(slot); self.labels.append(cvs); self.overlay_btns.append(btn)
            # Placeholder so the window appears fast
            draw_title(cvs, "Loading…")

        self._start_initial_prefetch()

    # ---------- Menus ----------
    def _open_gallery(self):
        try: GalleryWindow(self)
        except Exception: pass

    def _existing_norm_names(self) -> set[str]:
        from .persistence import normalize_term
        return {normalize_term(t.name) for t in self.terms}

    def add_terms_from_list(self, names: list[str]) -> tuple[int, list[str]]:
        existing = self._existing_norm_names(); added = 0; skipped: list[str] = []
        for raw in names:
            name = (raw or "").strip()
            if not name: continue
            from .persistence import normalize_term
            norm = normalize_term(name)
            if norm in existing: skipped.append(name); continue
            self.terms.append(Term(name)); existing.add(norm); added += 1
        try: save_terms(self.terms_path, self.terms)
        except Exception: pass
        return added, skipped

    def _open_add_terms(self):
        win = tk.Toplevel(self.root); win.title("Add Terms (one per line)"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Paste new terms, one per line. Existing ones will be skipped.").pack(anchor="w", pady=(0,6))
        txt = tk.Text(frm, width=50, height=16); txt.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(8,0))
        def on_add():
            raw = txt.get("1.0","end"); names = [s.strip() for s in raw.splitlines() if s.strip()]
            self.add_terms_from_list(names); win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear(); self._start_initial_prefetch()
        ttk.Button(btns, text="Add", command=on_add).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _open_remove_terms(self):
        win = tk.Toplevel(self.root); win.title("Remove Terms"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Select one or more terms to remove:").pack(anchor="w")
        lb = tk.Listbox(frm, selectmode=tk.EXTENDED, width=40, height=20); lb.pack(fill=tk.BOTH, expand=True, pady=6)
        sorted_terms = sorted(self.terms, key=lambda t: t.name.casefold())
        for t in sorted_terms: lb.insert(tk.END, f"{t.name}  ({t.rating:.0f}, g={t.games})")
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(8,0))
        def on_remove():
            sel = lb.curselection()
            if not sel: win.destroy(); return
            to_remove = [sorted_terms[i] for i in sel]
            names = [t.name for t in to_remove]
            self.terms = [t for t in self.terms if t not in to_remove]
            for nm in names:
                try: del self.used[nm]
                except Exception: pass
            try: save_terms(self.terms_path, self.terms); save_used(self.used_path, self.used)
            except Exception: pass
            win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear(); self._start_initial_prefetch()
        ttk.Button(btns, text="Remove Selected", command=on_remove).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _open_file(self):
        path = filedialog.askopenfilename(title="Select terms file", filetypes=[("Text/CSV","*.txt *.csv"),("All files","*.*")])
        if not path: return
        terms = load_terms(path)
        if not terms: messagebox.showerror("No Terms","That file is empty."); return
        self.terms_path = path; cfg = load_config(); cfg["last_terms_path"] = path; save_config(cfg)
        self.terms = terms
        self.used_path = used_cache_path(path); self.seen_hashes_path = seen_hashes_path(path)
        self.used = load_used(self.used_path); self.seen_hashes = load_seen_hashes(self.seen_hashes_path)
        self.current_four = []; self.current_urls = [None]*4; self.current_raw_bytes = [None]*4; self.current_pils = [None]*4
        self.recency_map.clear(); self.round_id = 0
        for lbl in self.labels: lbl.delete("all"); draw_title(lbl, "Loading…")
        with self.prefetch_lock: self.prefetch_queue.clear()
        self._start_initial_prefetch()

    def _open_settings(self):
        import termsranker.images as images
        win = tk.Toplevel(self.root); win.title("Settings"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        enforce_woman_var = tk.BooleanVar(value=images.ENFORCE_WOMAN)
        ddg_max_var = tk.IntVar(value=images.DDG_MAX_RESULTS)
        topk_var = tk.IntVar(value=images.TOP_SAMPLE_K)
        min_dim_var = tk.IntVar(value=images.MIN_DIM)
        timeout_var = tk.IntVar(value=images.TIMEOUT)
        def add_row(r, label, widget):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)
        frm.columnconfigure(1, weight=1); r=0
        add_row(r, "Require woman filter", ttk.Checkbutton(frm, variable=enforce_woman_var)); r+=1
        add_row(r, "DDG max results", ttk.Entry(frm, textvariable=ddg_max_var)); r+=1
        add_row(r, "Top sample K", ttk.Entry(frm, textvariable=topk_var)); r+=1
        add_row(r, "Minimum image dim", ttk.Entry(frm, textvariable=min_dim_var)); r+=1
        add_row(r, "HTTP timeout (s)", ttk.Entry(frm, textvariable=timeout_var)); r+=1
        def apply_and_close():
            images.ENFORCE_WOMAN = bool(enforce_woman_var.get())
            try: images.DDG_MAX_RESULTS = max(10, int(ddg_max_var.get()))
            except Exception: pass
            try: images.TOP_SAMPLE_K = max(1, int(topk_var.get()))
            except Exception: pass
            try: images.MIN_DIM = max(0, int(min_dim_var.get()))
            except Exception: pass
            try: images.TIMEOUT = max(1, int(timeout_var.get()))
            except Exception: pass
            win.destroy()
        ttk.Button(frm, text="OK", command=apply_and_close).grid(row=r, column=1, sticky="e", pady=(8,0))

    # ---------- Prefetch orchestration ----------
    def _start_initial_prefetch(self):
        # Only one batch to reduce startup delay
        def worker():
            batch = self._prefetch_round()
            self.root.after(0, lambda: self._display_batch(batch))
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
        threading.Thread(target=worker, daemon=True).start()

    def _top_up_prefetch_async(self):
        try:
            while True:
                batch = self._prefetch_round()
                with self.prefetch_lock:
                    if len(self.prefetch_queue) >= 2: break
                    self.prefetch_queue.append(batch)
        except Exception:
            pass

    def _consume_prefetch_or_fetch(self) -> PrefetchBatch:
        with self.prefetch_lock:
            batch = self.prefetch_queue.pop(0) if self.prefetch_queue else None
        if batch and len(batch.entries) == 4:
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
            return batch
        return self._prefetch_round()

    # ---------- Candidate picking (NO UI in worker) ----------
    def _prefetch_round(self) -> PrefetchBatch:
        if len(self.terms) < 4: return PrefetchBatch(entries=[])

        # sample 4 with recency cooldown
        four = weighted_sample_terms(self.terms, 4, recency_map=self.recency_map)

        extra = self.extra_var.get()
        round_hosts: set[str] = set()
        used_sets = [self.used.get(t.name, set()) for t in four]
        urls: list[str | None] = [None]*4
        datas: list[bytes | None] = [None]*4
        hashes: list[str | None] = [None]*4

        # primary pass
        for i, term in enumerate(four):
            candidates = ddg_candidates(term.name, extra)
            if candidates:
                for u in candidates:
                    if u in used_sets[i]: continue
                    hst = host_of(u)
                    if hst and hst in round_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    hh = phash_bytes(d)
                    if hh and hh in self.seen_hashes: continue
                    urls[i], datas[i], hashes[i] = u, d, hh
                    round_hosts.add(hst)
                    if term.name not in self.used: self.used[term.name] = set()
                    self.used[term.name].add(u)
                    if hh: self.seen_hashes.add(hh)
                    break

        # fallback: swap in alternate terms for failed slots
        try:
            exclude = set(t.name for t in four)
            for i in range(4):
                if datas[i] is not None: continue
                alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
                alt = alt_list[0] if alt_list else None
                if not alt: continue
                exclude.add(alt.name)
                used_set = self.used.get(alt.name, set())
                cand = ddg_candidates(alt.name, extra)
                url = None; data = None; hh = None
                if cand:
                    for u in cand:
                        if u in used_set: continue
                        hst = host_of(u)
                        if hst and hst in round_hosts: continue
                        d = download_image_bytes(u, u)
                        if not d: continue
                        if not looks_like_woman(d): continue
                        hh2 = phash_bytes(d)
                        if hh2 and hh2 in self.seen_hashes: continue
                        url, data, hh = u, d, hh2
                        round_hosts.add(hst); break
                urls[i], datas[i], hashes[i] = url, data, hh
                if url:
                    if alt.name not in self.used: self.used[alt.name] = set()
                    self.used[alt.name].add(url)
                    if hh: self.seen_hashes.add(hh)
                    four[i] = alt
        except Exception:
            pass

        # persist caches best-effort
        try:
            save_used(self.used_path, self.used); save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
        except Exception:
            pass

        # recency update
        for k in list(self.recency_map.keys()):
            self.recency_map[k] = min(self.recency_map[k] + 1, 10 * COOLDOWN_WINDOW if COOLDOWN_WINDOW else 999999)
        for t in four: self.recency_map[t.name] = 0
        self.round_id += 1

        # Build atomic entries
        entries = [PrefetchEntry(term=four[i], url=urls[i], data=datas[i]) for i in range(4)]
        return PrefetchBatch(entries=entries)

    # ---------- Display (main thread only) ----------
    def _display_batch(self, batch: PrefetchBatch):
        if not batch.entries or len(batch.entries) != 4:
            messagebox.showerror("Need more terms","At least 4 terms are required."); return

        # Atomically replace current state from the entries list
        self.current_four = [e.term for e in batch.entries]
        self.current_urls = [e.url for e in batch.entries]
        self.current_raw_bytes = [e.data for e in batch.entries]
        self.current_pils = [None]*4

        for i, entry in enumerate(batch.entries):
            term = entry.term
            lbl = self.labels[i]
            draw_title(lbl, term.name)
            if DEBUG_OVERLAY:
                host = host_of(entry.url) if entry.url else "-"
                draw_debug(lbl, host)
            data = entry.data
            if data:
                try:
                    pil = Image.open(io.BytesIO(data)).convert("RGB")
                    self.current_pils[i] = pil
                    render_to_canvas(lbl, pil)
                    draw_title(lbl, term.name)
                    if DEBUG_OVERLAY:
                        draw_debug(lbl, host_of(entry.url) if entry.url else "-")
                except Exception:
                    self.current_pils[i] = None
                    canvas_show_error(lbl, "Failed to load"); draw_title(lbl, term.name)
            else:
                self.current_pils[i] = None
                canvas_show_error(lbl, "No image"); draw_title(lbl, term.name)

    # ---------- Events ----------
    def _on_canvas_resize(self, idx: int):
        try:
            pil = self.current_pils[idx]; cvs = self.labels[idx]
            if pil is not None: render_to_canvas(cvs, pil)
            if self.current_four and idx < len(self.current_four):
                draw_title(cvs, self.current_four[idx].name)
                if DEBUG_OVERLAY:
                    draw_debug(cvs, host_of(self.current_urls[idx]) if self.current_urls[idx] else "-")
        except Exception: pass

    def _on_click(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        winner = self.current_four[idx]; losers = [t for i, t in enumerate(self.current_four) if i != idx]
        # Save winning image (best-effort)
        self._save_winning_image_uncropped(idx, winner)
        # Update ratings
        elo_update(winner, losers)
        try: save_terms(self.terms_path, self.terms)
        except Exception: pass
        # Fetch and show a fresh, coherent batch
        batch = self._consume_prefetch_or_fetch()
        self._display_batch(batch)

    def _on_other(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        old_term = self.current_four[idx]; extra = self.extra_var.get()
        def worker():
            self.used[old_term.name] = set()
            try: save_used(self.used_path, self.used)
            except Exception: pass

            exclude = {t.name for i, t in enumerate(self.current_four) if i != idx}
            alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
            new_term = alt_list[0] if alt_list else None
            if new_term is None: return

            other_hosts = {host_of(u) for i, u in enumerate(self.current_urls) if i != idx and u}
            url = None; data = None; hh = None
            candidates = ddg_candidates(new_term.name, extra)

            def try_candidates(cands, respect_hosts=True):
                nonlocal url, data, hh
                if not cands: return False
                used_set = self.used.get(new_term.name, set())
                for u in cands:
                    if u in used_set: continue
                    if respect_hosts:
                        hst = host_of(u)
                        if hst and hst in other_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    hhash = phash_bytes(d)
                    if hhash and hhash in self.seen_hashes: continue
                    url, data, hh = u, d, hhash
                    return True
                return False

            ok = try_candidates(candidates, True)
            if not ok: ok = try_candidates(candidates, False)

            if url:
                if new_term.name not in self.used: self.used[new_term.name] = set()
                self.used[new_term.name].add(url)
                try: save_used(self.used_path, self.used)
                except Exception: pass
            if hh:
                self.seen_hashes.add(hh)
                try: save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
                except Exception: pass

            def apply():
                self.current_four[idx] = new_term
                self.current_urls[idx] = url
                self.current_raw_bytes[idx] = data
                lbl = self.labels[idx]
                draw_title(lbl, new_term.name)
                if DEBUG_OVERLAY:
                    draw_debug(lbl, host_of(url) if url else "-")
                if data:
                    try:
                        pil = Image.open(io.BytesIO(data)).convert("RGB")
                        self.current_pils[idx] = pil
                        render_to_canvas(lbl, pil)
                        draw_title(lbl, new_term.name)
                        if DEBUG_OVERLAY:
                            draw_debug(lbl, host_of(url) if url else "-")
                    except Exception:
                        self.current_pils[idx] = None
            self.root.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Save winning image ----------
    def _save_winning_image_uncropped(self, idx: int, term: Term):
        data = self.current_raw_bytes[idx]
        if not data: return
        term_dir = os.path.join(os.path.dirname(os.path.abspath(self.terms_path)), "ranked_images", slugify(term.name))
        os.makedirs(term_dir, exist_ok=True)
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{slugify(term.name)}_{int(round(term.rating))}_{ts}.jpg"
        path = os.path.join(term_dir, fname)
        try: pil.save(path, format="JPEG", quality=90)
        except Exception: pass

# -------------------- Entry --------------------
def main():
    arg_path = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk(); root.geometry("1300x1000"); root.minsize(900, 700)
    App(root, arg_path); root.mainloop()

# -------------------- Gallery (unchanged core behavior) --------------------
class GalleryWindow(tk.Toplevel):
    BATCH_SIZE = 8
    VIS_BUFFER = 800

    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.title("All Ranked Images"); self.geometry("1200x800")
        self.app = app
        self._images: list[ImageTk.PhotoImage] = []
        self.thumb_height_var = tk.IntVar(value=200)  # default height
        self.sort_mode = tk.StringVar(value="rating_desc")
        self._thumb_cache: dict[tuple[str, int], ImageTk.PhotoImage] = {}
        self._row_widgets: list[tuple[ttk.Frame, dict]] = []

        container = ttk.Frame(self); container.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(container); top.pack(fill=tk.X, padx=10, pady=(8,0))
        ttk.Label(top, text="Height:").pack(side=tk.LEFT)
        for s in [50,150,200,250,300]:
            ttk.Button(top, text=str(s), command=lambda size=s: self._set_height(size)).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="   Sort:").pack(side=tk.LEFT, padx=(12,0))
        sort_bar = ttk.Frame(top); sort_bar.pack(side=tk.LEFT)
        def add_sort(label, key):
            ttk.Button(sort_bar, text=label, command=lambda k=key: self._set_sort(k)).pack(side=tk.LEFT, padx=2)
        add_sort("Alphabetical", "name_asc")
        add_sort("Highest Rating", "rating_desc")
        add_sort("Lowest Rating", "rating_asc")
        add_sort("Games High", "games_desc")
        add_sort("Games Low", "games_asc")
        add_sort("Sigma High", "sigma_desc")
        add_sort("Sigma Low", "sigma_asc")
        ttk.Button(top, text="Refresh", command=self._refresh).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg="#111")
        vbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = ttk.Frame(self.canvas); self.inner_id = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_frame_configure); self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_scroll, add="+")
        self.canvas.bind("<Button-4>", self._on_scroll); self.canvas.bind("<Button-5>", self._on_scroll)

        self._build_rows_shells()

    def _set_height(self, size: int):
        self.thumb_height_var.set(size); self._refresh()

    def _set_sort(self, mode: str):
        self.sort_mode.set(mode); self._refresh()

    def _on_frame_configure(self, event=None): self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width); self._ensure_visible_rows()

    def _on_scroll(self, event=None): self.after(10, self._ensure_visible_rows())

    def _refresh(self):
        for w in list(self.inner.children.values()):
            try: w.destroy()
            except Exception: pass
        self._images.clear(); self._row_widgets.clear()
        self._build_rows_shells()

    def _sorted_terms(self):
        key = self.sort_mode.get()
        terms = list(self.app.terms)
        if key == "name_asc": terms.sort(key=lambda t: t.name.casefold())
        elif key == "rating_asc": terms.sort(key=lambda t: t.rating)
        elif key == "rating_desc": terms.sort(key=lambda t: t.rating, reverse=True)
        elif key == "games_asc": terms.sort(key=lambda t: t.games)
        elif key == "games_desc": terms.sort(key=lambda t: t.games, reverse=True)
        elif key == "sigma_asc": terms.sort(key=lambda t: t.sigma)
        elif key == "sigma_desc": terms.sort(key=lambda t: t.sigma, reverse=True)
        else: terms.sort(key=lambda t: t.rating, reverse=True)
        return terms

    def _build_rows_shells(self):
        base_dir = os.path.dirname(os.path.abspath(self.app.terms_path))
        ranked_dir = os.path.join(base_dir, "ranked_images")
        terms_sorted = self._sorted_terms()

        for term in terms_sorted:
            term_dir = os.path.join(ranked_dir, slugify(term.name))
            if not os.path.isdir(term_dir): continue
            files = sorted([f for f in Path(term_dir).glob("*.jpg")], key=lambda p: p.name, reverse=True)
            if not files: continue

            row_frame = ttk.Frame(self.inner); row_frame.pack(fill=tk.X, padx=10, pady=(12, 4))
            header = ttk.Frame(row_frame); header.pack(fill=tk.X)
            stats = f"rating {term.rating:.0f} | games {term.games} | sigma {term.sigma:.1f} | images {len(files)}"
            ttk.Label(header, text=f"{term.name} — {stats}", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, anchor="w")
            more_btn_holder = ttk.Frame(header); more_btn_holder.pack(side=tk.RIGHT)
            thumbs_wrap = ttk.Frame(row_frame); thumbs_wrap.pack(fill=tk.X, padx=0, pady=(4, 0))

            meta = {"term": term, "dir": term_dir, "files": files, "wrap": thumbs_wrap,
                    "start": 0, "built": False, "more_btn_holder": more_btn_holder}
            self._row_widgets.append((row_frame, meta))

        self.after(0, self._ensure_visible_rows)

    def _ensure_visible_rows(self):
        try:
            top = int(self.canvas.canvasy(0)); bottom = int(self.canvas.canvasy(self.canvas.winfo_height()))
        except Exception: return
        top -= 800; bottom += 800
        for row_frame, meta in self._row_widgets:
            try:
                ry0 = row_frame.winfo_y(); ry1 = ry0 + row_frame.winfo_height()
            except Exception: continue
            if ry1 < top or ry0 > bottom: continue
            if not meta.get("built"): self._build_row_content(meta)

    def _build_row_content(self, meta: dict):
        files = meta["files"]; wrap: ttk.Frame = meta["wrap"]
        start = 0; page = 8; subset = files[start:start+page]
        self._render_thumbs_async(wrap, subset)
        meta["start"] = start + page; meta["built"] = True
        if len(files) > page:
            holder: ttk.Frame = meta["more_btn_holder"]
            btn = ttk.Button(holder, text="Load more…"); btn.pack()
            btn.configure(command=lambda m=meta, b=btn: self._on_load_more(m, b))

    def _on_load_more(self, meta: dict, btn: ttk.Button | None):
        files = meta["files"]; start = meta.get("start", 0); page = 8
        subset = files[start:start+page]
        self._render_thumbs_async(meta["wrap"], subset)
        start += page; meta["start"] = start
        if start >= len(files) and btn is not None:
            try: btn.destroy()
            except Exception: pass

    def _render_thumbs_async(self, parent: ttk.Frame, file_list: list[Path]):
        if not file_list: return
        target_h = int(self.thumb_height_var.get())
        def worker():
            items = []
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.ANTIALIAS  # type: ignore
            for fp in file_list:
                try:
                    key = (str(fp), target_h)
                    if key in self._thumb_cache:
                        tkimg = self._thumb_cache[key]; items.append((fp, tkimg)); continue
                    pil = Image.open(fp).convert("RGB")
                    ratio = target_h / max(1, pil.height)
                    new_w = max(1, int(round(pil.width * ratio)))
                    thumb_img = pil.resize((new_w, target_h), resample=resample)
                    tkimg = ImageTk.PhotoImage(thumb_img)
                    if len(self._thumb_cache) > 800:
                        try: self._thumb_cache.pop(next(iter(self._thumb_cache)))
                        except Exception: pass
                    self._thumb_cache[key] = tkimg
                    items.append((fp, tkimg))
                except Exception:
                    continue
            def apply():
                c = 0
                for fp, tkimg in items:
                    lbl = tk.Label(parent, image=tkimg, bg="#111", cursor="hand2")
                    lbl.image = tkimg
                    lbl.bind("<Button-1>", lambda e, path=str(fp): self._open_fullscreen(path))
                    lbl.grid(row=0, column=c, padx=6, pady=6); c += 1
                    self._images.append(tkimg)
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _open_fullscreen(self, path: str):
        FullscreenImageWindow(self, path)

class FullscreenImageWindow(tk.Toplevel):
    def __init__(self, parent: GalleryWindow, path: str):
        super().__init__(parent)
        self.title(os.path.basename(path)); self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.destroy()); self.bind("<Button-1>", lambda e: self.destroy())
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)
        self._pil = Image.open(path).convert("RGB"); self._tk = None
        self.bind("<Configure>", self._on_resize); self._on_resize()
    def _on_resize(self, event=None):
        w = self.winfo_width(); h = self.winfo_height()
        if w <= 2 or h <= 2: return
        try: resample = Image.Resampling.LANCZOS
        except Exception: resample = Image.ANTIALIAS  # type: ignore
        fitted = ImageOps.contain(self._pil, (w, h), method=resample)
        self._tk = ImageTk.PhotoImage(fitted)
        self.canvas.delete("img"); self.canvas.create_image(w//2, h//2, image=self._tk, anchor="center", tags="img")
