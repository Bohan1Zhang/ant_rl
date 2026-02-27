from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np

# 4-direction movement (Up, Down, Left, Right)
DIRS4: List[Tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def random_empty_cell(
    rng: np.random.Generator,
    obstacles: np.ndarray,
    forbidden: set[Tuple[int, int]],
) -> Tuple[int, int]:
    """Pick a random empty cell not in forbidden."""
    h, w = obstacles.shape
    for _ in range(2000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if obstacles[y, x] == 0 and (x, y) not in forbidden:
            return x, y

    # Fallback scan (deterministic)
    for y in range(h):
        for x in range(w):
            if obstacles[y, x] == 0 and (x, y) not in forbidden:
                return x, y
    # If everything is blocked, return a default safe
    return 0, 0