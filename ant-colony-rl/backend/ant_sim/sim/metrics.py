from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class WorldMetrics:
    delivered_by_colony: List[int]

    def reset(self, num_colonies: int) -> None:
        self.delivered_by_colony = [0 for _ in range(num_colonies)]