import argparse
from .rating import rate_terms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SearchTermRating",
        description="Rate and analyze search terms.",
    )
    parser.add_argument(
        "terms",
        nargs="*",
        help="One or more search terms to rate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Filter to terms with score >= threshold (default: 0.5)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.terms:
        parser.print_help()
        return

    results = rate_terms(args.terms)
    for term, score in results:
        if score >= args.threshold:
            print(f"{term}\t{score:.3f}")
