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

from src.config_parser import load_config
from src.orchestrator import Orchestrator


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
        cfg.mock.enabled = True
        for m in cfg.models:
            m.backend = "mock"

    orchestrator = Orchestrator(cfg)
    out_path = orchestrator.run()
    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
