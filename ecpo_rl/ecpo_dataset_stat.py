# -*- coding: utf-8 -*-
"""CLI entrypoint for ECPO dataset statistics."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ecpo_rl.dataset_stat import main  # type: ignore
else:
    from .dataset_stat import main


if __name__ == "__main__":
    main()
