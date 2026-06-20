#!/usr/bin/env python3
"""Entry point: python run_sweep.py [--config config/matrix_search.yaml] [--mock]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).parent))

from src.config_parser import BenchmarkConfig, load_config  # noqa: E402 — must follow sys.path setup above
from src.orchestrator import Orchestrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Local Benchmarking Matrix Sweep")
    parser.add_argument(
        "--config", default="config/matrix_search.yaml",
        help="Path to matrix_search.yaml",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Force mock mode (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        # Re-validate rather than mutating cfg.mock.enabled + each model's
        # backend by hand — that duplicated BenchmarkConfig's own
        # _mock_overrides_backends validator and would silently drift out of
        # sync if that validator's logic ever changed.
        raw = cfg.model_dump()
        raw["mock"]["enabled"] = True
        cfg = BenchmarkConfig.model_validate(raw)

    orchestrator = Orchestrator(cfg)
    out_path = orchestrator.run()
    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
