"""Command-line interface for the QUALIPHIDE inference pipeline."""

import argparse

from qualiphide.pipeline import run_pipeline


def main():
    """Entry point for the ``qualiphide`` console script."""
    parser = argparse.ArgumentParser(description="QUALIPHIDE inference pipeline")
    parser.add_argument(
        "config_name",
        nargs="?",
        default="config",
        help="Config name (without .yaml extension; default: config)",
    )
    parser.add_argument(
        "--keep-toymc",
        action="store_true",
        help="Keep all ToyMC datasets instead of compacting to 50 per file "
             "(sensitivity mode only)",
    )
    parser.add_argument(
        "--mode",
        choices=["sensitivity", "coverage"],
        default="sensitivity",
        help="Pipeline mode (default: sensitivity)",
    )
    args = parser.parse_args()

    if args.mode == "coverage":
        from qualiphide.coverage import run_coverage
        run_coverage(args.config_name)
    else:
        run_pipeline(args.config_name, keep_toymc=args.keep_toymc)
