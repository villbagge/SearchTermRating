from __future__ import annotations
import argparse
import tkinter as tk

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="SearchTermRating", description="Rate and analyze search terms.")
    sub = p.add_subparsers(dest="cmd", required=False)

    # Text-mode demo
    p_rank = sub.add_parser("rank", help="Rank terms in text mode (no GUI)")
    p_rank.add_argument("terms", nargs="*", help="Search terms")
    p_rank.add_argument("--threshold", type=float, default=0.5, help="Filter score >= threshold")

    # GUI launcher — path optional now
    p_gui = sub.add_parser("gui", help="Launch the Tk GUI to compare images")
    p_gui.add_argument("--terms-file", dest="terms_file", default="",
                       help="Optional: path to a TXT/CSV with terms (GUI will prompt if omitted)")
    return p

def main(argv: list[str] | None = None) -> None:
    from . import rating as rating_mod
    from . import gui_app

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == "rank":
        if not args.terms:
            parser.print_help()
            return
        results = rating_mod.rate_terms(args.terms)
        for term, score in results:
            if score >= args.threshold:
                print(f"{term}\t{score:.3f}")
        return

    if args.cmd == "gui":
        initial_path = args.terms_file or None  # let GUI handle prompt/persist
        root = tk.Tk()
        root.geometry("1300x1000")
        root.minsize(900, 700)
        gui_app.App(root, initial_path)
        root.mainloop()
        return
