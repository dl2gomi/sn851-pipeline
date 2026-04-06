import argparse
from pathlib import Path
import sys

from ..config import load_app_config
from .orchestrator import PipelineOrchestrator


def parse_args() -> argparse.Namespace:
    _ensure_backward_compatible_run_subcommand()
    parser = argparse.ArgumentParser(description="SN851 RL optimization pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configuration.yaml"),
        help="Path to pipeline configuration.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run pipeline orchestrator")
    run.add_argument("--steps", type=int, default=None, help="Number of orchestrator steps")
    run.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    run.add_argument("--rollouts-per-step", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--model-dir", type=Path, default=None, help="Local model folder containing config.json")
    run.add_argument("--train-all-sampling-envs", action="store_true")
    run.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable id for this training run (Postgres/Prometheus); default: random UUID",
    )

    return parser.parse_args()


def _ensure_backward_compatible_run_subcommand() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("run")
        return
    if "run" in sys.argv[1:]:
        return
    if sys.argv[1] == "--config":
        insert_pos = 3 if len(sys.argv) >= 3 else 2
        sys.argv.insert(insert_pos, "run")
        return
    if sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")


def run_cli() -> None:
    args = parse_args()
    app_cfg = load_app_config(args.config)
    cfg = app_cfg.pipeline
    if args.rollouts_per_step is not None:
        cfg.rollouts_per_step = args.rollouts_per_step
        cfg.training.rollout_batch_size = args.rollouts_per_step
    if args.batch_size is not None:
        cfg.training.mini_batch_size = args.batch_size
    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.run_id is not None:
        cfg.run_id = args.run_id.strip() or None
    cfg.train_scoring_envs_only = not args.train_all_sampling_envs
    orch = PipelineOrchestrator(cfg)
    steps = args.steps if args.steps is not None else app_cfg.cli.run.steps
    dry_run = args.dry_run or app_cfg.cli.run.dry_run
    orch.run(steps=steps, dry_run=dry_run)
