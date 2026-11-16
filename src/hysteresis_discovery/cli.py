"""Command-line entry point for all paper experiments."""

import argparse
from dataclasses import asdict, fields
import importlib
import json
from pathlib import Path

from .registry import EXPERIMENTS
from .runtime import RunConfig


def make_parser(defaults=None):
    parser = argparse.ArgumentParser(
        description="Equation discovery and dynamic prediction of hysteretic systems."
    )
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS), nargs="?")
    parser.add_argument("--stage", choices=("step12", "step3", "all"), default="step12")
    parser.add_argument("--config", type=Path, help="JSON file with runtime defaults.")
    parser.add_argument(
        "--list", action="store_true", help="List available experiments."
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Check required inputs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the run plan and exit."
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--niterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dpi", type=int)
    parser.add_argument(
        "--plot-interval",
        type=int,
        help="Checkpoint frequency in epochs; 0 disables intermediate figures and NPZs.",
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction)
    parser.add_argument("--usetex", action=argparse.BooleanOptionalAction)
    parser.set_defaults(**(asdict(RunConfig()) if defaults is None else defaults))
    return parser


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    defaults = asdict(RunConfig())
    parser = make_parser()
    if selected.config:
        try:
            values = json.loads(selected.config.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("The config must be a JSON object.")
            unknown = set(values) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
            defaults.update(values)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    parser = make_parser(defaults)
    args = parser.parse_args(argv)
    for name in ("epochs", "niterations", "dpi"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            parser.error(f"{name} must be a positive integer.")
    if args.plot_interval is not None and (
        type(args.plot_interval) is not int or args.plot_interval < 0
    ):
        parser.error("plot_interval must be a non-negative integer.")
    if args.seed is not None and (
        type(args.seed) is not int or not 0 <= args.seed < 2**32
    ):
        parser.error("seed must be an integer between 0 and 2**32 - 1.")
    if args.device not in ("auto", "cpu", "cuda"):
        parser.error("device must be auto, cpu, or cuda.")
    for name in ("show", "usetex"):
        if type(getattr(args, name)) is not bool:
            parser.error(f"{name} must be a boolean.")
    for name in ("data_root", "output_root"):
        value = getattr(args, name)
        if not isinstance(value, (str, Path)) or not str(value):
            parser.error(f"{name} must be a non-empty path.")
        setattr(args, name, Path(value).resolve())
    return parser, args


def check_data(experiment, data_root):
    """Report missing inputs before importing training or Julia dependencies."""
    for filename in EXPERIMENTS[experiment].files:
        path = data_root / experiment / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing input data: {path}")


def main(argv=None):
    parser, args = parse_args(argv)
    if args.list:
        for name, spec in EXPERIMENTS.items():
            print(f"{name:24s} Section {spec.section}: {spec.title}")
        return 0
    if not args.experiment and not args.check_data:
        parser.error("Choose an experiment, or use --list / --check-data.")
    config = RunConfig(
        **{field.name: getattr(args, field.name) for field in fields(RunConfig)}
    )
    names = [args.experiment] if args.experiment else list(EXPERIMENTS)
    try:
        if args.check_data:
            for name in names:
                check_data(name, config.data_root)
                print(f"[OK] {name}")
            return 0
        stages = ("step12", "step3") if args.stage == "all" else (args.stage,)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "experiment": args.experiment,
                        "stages": stages,
                        "config": asdict(config),
                        "output_directory": config.output_root / args.experiment,
                    },
                    default=str,
                    indent=2,
                )
            )
            return 0
        for stage in stages:
            if stage == "step12":
                check_data(args.experiment, config.data_root)
            else:
                checkpoint = config.output_root / args.experiment / "data_Final.npz"
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Missing Step 2 result: {checkpoint}. Run --stage step12 first."
                    )
            module = importlib.import_module(
                f"hysteresis_discovery.experiments.{args.experiment}.{stage}"
            )
            module.run(config)
    except (OSError, ModuleNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0
