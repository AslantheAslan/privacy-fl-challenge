#!/usr/bin/env python3
"""Challenge entry point.

    python run_submission.py \
      --train data/train.jsonl \
      --input data/validation_inputs.jsonl \
      --output outputs/validation_predictions.jsonl \
      --artifacts-dir outputs/artifacts

Optional flags (not needed by the evaluator):
  --eval-labels PATH   ground truth for --input; adds holdout metrics to the summary
  --quick              fewer seeds/repeats for the analysis battery (predictions unchanged)
  --seed INT           seed of the submitted federated model (default 7)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import configure_logging, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--eval-labels", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    configure_logging()
    log = logging.getLogger("submission")
    result = run(args.train, args.input, args.output, args.artifacts_dir, args.eval_labels, args.quick, args.seed)
    log.info("wrote %d predictions to %s in %.1fs", result["predictions"], args.output, result["timings"]["total_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
