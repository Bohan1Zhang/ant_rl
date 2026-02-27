from __future__ import annotations

import numpy as np


def diffuse4(pher: np.ndarray, obstacles: np.ndarray, diff_rate: float) -> np.ndarray:
    """
    Simple 4-neighbor diffusion without wrap-around.
    pher: (H, W), obstacles: (H, W) with 1=blocked
    """
    if diff_rate <= 0.0:
        out = pher.copy()
        out[obstacles == 1] = 0.0
        return out

    padded = np.pad(pher, 1, mode="constant", constant_values=0.0)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    avg = (up + down + left + right) * 0.25

    out = (1.0 - diff_rate) * pher + diff_rate * avg
    out[obstacles == 1] = 0.0
    return out


def evaporate(pher: np.ndarray, evap_rate: float) -> np.ndarray:
    if evap_rate <= 0.0:
        return pher
    out = pher * (1.0 - evap_rate)
    out[out < 1e-7] = 0.0
    return out