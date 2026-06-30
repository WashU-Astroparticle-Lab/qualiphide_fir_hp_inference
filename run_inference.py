"""Backward-compatible CLI entry point for the QUALIPHIDE pipeline.

Prefer ``qualiphide <config_name>`` (installed via ``pip install -e .``)
or ``python -m qualiphide.cli <config_name>`` for new usage.
"""

from qualiphide.cli import main

if __name__ == "__main__":
    main()
