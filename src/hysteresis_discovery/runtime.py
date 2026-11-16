"""Paths, logging, plotting, and run metadata shared by all experiments."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

from .registry import PROJECT_ROOT


@dataclass(frozen=True)
class RunConfig:
    """Runtime overrides; scientific parameters remain in each experiment."""

    data_root: Path = PROJECT_ROOT / "data"
    output_root: Path = PROJECT_ROOT / "outputs"
    epochs: int = 10000
    niterations: int = 2000
    seed: int | None = None
    device: str = "auto"
    show: bool = False
    usetex: bool = False
    dpi: int = 600
    plot_interval: int | None = None


class Tee:
    """Write output to the active console and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@dataclass
class Session:
    """Resolved paths and device for one experiment stage."""

    data_dir: Path
    output_dir: Path
    device: object
    symbolic_fit_count: int = 0

    def save_equations(self, **equations):
        """Export discovered equations in a readable, machine-friendly form."""
        destination = self.output_dir / "equations.json"
        destination.write_text(
            json.dumps({name: str(expr) for name, expr in equations.items()}, indent=2),
            encoding="utf-8",
        )

    def symbolic_options(self, config):
        """Keep each PySR search under this experiment's output directory."""
        self.symbolic_fit_count += 1
        options = {
            "output_directory": str(self.output_dir / "symbolic"),
            "run_id": f"fit_{self.symbolic_fit_count:02d}",
        }
        if config.seed is not None:
            options.update(
                random_state=config.seed,
                deterministic=True,
                parallelism="serial",
            )
        return options

    def checkpoint(self, epoch, default_interval, include_first=False, interval=None):
        """Decide when to save intermediate plots and trajectory snapshots."""
        if interval == 0:
            return False
        frequency = default_interval if interval is None else interval
        return (include_first and epoch == 0) or (epoch + 1) % frequency == 0


@contextmanager
def experiment_session(config, experiment, stage):
    """Configure a run and restore process state even when an experiment fails."""
    import matplotlib
    import numpy as np
    import torch

    if not config.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(config.output_root).resolve() / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_root).resolve() / experiment
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")

    metadata = {
        "experiment": experiment,
        "stage": stage,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "resolved_device": str(device),
        "python": sys.version,
        "packages": {},
        "status": "running",
    }
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "torchdiffeq",
        "pysr",
        "sympy",
        "scikit-learn",
    ):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata_path = output_dir / f"run_{stage}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    previous_dtype = torch.get_default_dtype()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    original_show, original_savefig = plt.show, plt.savefig

    def show_or_close(*args, **kwargs):
        if config.show:
            original_show(*args, **kwargs)
        else:
            plt.close("all")

    def savefig(*args, **kwargs):
        kwargs["dpi"] = config.dpi
        return original_savefig(*args, **kwargs)

    try:
        with (
            torch.random.fork_rng(),
            plt.rc_context({"text.usetex": config.usetex, "font.family": "serif"}),
            (output_dir / f"log_{stage}.txt").open("w", encoding="utf-8") as logfile,
        ):
            plt.show, plt.savefig = show_or_close, savefig
            if config.seed is not None:
                random.seed(config.seed)
                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.seed)
            torch.set_default_dtype(torch.float64)
            with (
                redirect_stdout(Tee(sys.stdout, logfile)),
                redirect_stderr(Tee(sys.stderr, logfile)),
            ):
                print(f"Experiment: {experiment} | Stage: {stage} | Device: {device}")
                print(f"Data: {data_dir}\nOutputs: {output_dir}")
                yield Session(data_dir, output_dir, device)
            metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        plt.show, plt.savefig = original_show, original_savefig
        plt.close("all")
        torch.set_default_dtype(previous_dtype)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
