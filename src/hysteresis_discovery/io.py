"""Numeric checkpoint loading with explicit resource cleanup."""

from pathlib import Path

import numpy as np


def load_npz(path):
    """Load numeric arrays and close the archive before returning them."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {Path(path).name}, array '{key}'.")
    return arrays
