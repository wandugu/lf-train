# -*- coding: utf-8 -*-
"""CLI entrypoint for ECPO evaluation."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ecpo_rl.evaluate_ecpo_impl import main  # type: ignore
else:
    from .evaluate_ecpo_impl import main


if __name__ == "__main__":
    main()
