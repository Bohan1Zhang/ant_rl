from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np

from ant_sim.sim.entities import WorldConfig, Ant, Colony
from ant_sim.sim.utils import DIRS4, in_bounds, manhattan, random_empty_cell
from ant_sim.sim.pheromone import diffuse4, evaporate


class AntColonyWorld:
    """
    Grid world simulation:
    - Multiple colonies competing for multiple food sources
    - Food sources can deplete, respawn, and move
    - Pheromone field per colony (one channel per colony for simplicity)
    - Ants have local behavior; can be controlled externally (RL) per ant
    """

    def __init__(self, config: WorldConfig, seed: Optional[int] = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.obstacles = np.zeros((self.cfg.grid_h, self.cfg.grid_w), dtype=np.uint8)

        # per-colony pheromone map: (C, H, W)
        self.pheromone = np.zeros(
            (self.cfg.num_colonies, self.cfg.grid_h, self.cfg.grid_w),
            dtype=np.float32,
        )

        # foods: (x,y) -> amount
        self.foods: Dict[Tuple[int, int], int] = {}

        self.colonies: List[Colony] = []
        self.ants: List[Ant] = []

        # ant occupancy per colony: (C, H, W) int16
        self.ant_counts = np.zeros(
            (self.cfg.num_colonies, self.cfg.grid_h, self.cfg.grid_w),
            dtype=np.int16,
        )

        self.tick: int = 0

        self.reset(seed=seed)

    @property
    def num_ants(self) -> int:
        return len(self.ants)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.tick = 0

        # obstacles
        self.obstacles = (self.rng.random((self.cfg.grid_h, self.cfg.grid_w)) < self.cfg.obstacle_ratio).astype(np.uint8)

        # reset pheromone
        self.pheromone.fill(0.0)

        # nests (fixed corners style, easy and deterministic)
        self.colonies = []
        nests: List[Tuple[int, int]] = []
        for c in range(self.cfg.num_colonies):
            if c == 0:
                nx, ny = 5, 5
            elif c == 1:
                nx, ny = self.cfg.grid_w - 6, self.cfg.grid_h - 6
            else:
                # spread extra colonies
                nx = int((c + 1) * self.cfg.grid_w / (self.cfg.num_colonies + 1))
                ny = int((c + 1) * self.cfg.grid_h / (self.cfg.num_colonies + 1))
            nx = int(np.clip(nx, 0, self.cfg.grid_w - 1))
            ny = int(np.clip(ny, 0, self.cfg.grid_h - 1))
            nests.append((nx, ny))
            self.obstacles[ny, nx] = 0  # ensure nest cell is free
            self.colonies.append(Colony(colony_id=c, nest=(nx, ny), delivered=0))

        # foods
        self.foods = {}
        forbidden = set(nests)
        for _ in range(self.cfg.num_food_init):
            x, y = random_empty_cell(self.rng, self.obstacles, forbidden | set(self.foods.keys()))
            self.foods[(x, y)] = int(self.cfg.food_amount_init)

        # ants
        self.ants = []
        ant_id = 0
        for c in range(self.cfg.num_colonies):
            nx, ny = self.colonies[c].nest
            for _ in range(self.cfg.ants_per_colony):
                self.ants.append(Ant(ant_id=ant_id, colony_id=c, x=nx, y=ny))
                ant_id += 1

        # occupancy
        self.ant_counts.fill(0)
        for ant in self.ants:
            self.ant_counts[ant.colony_id, ant.y, ant.x] += 1

    # --------------------------
    # Food dynamics
    # --------------------------
    def _move_foods(self) -> None:
        if not self.foods:
            return
        moved: Dict[Tuple[int, int], int] = {}
        for (x, y), amt in self.foods.items():
            if amt <= 0:
                continue
            if self.rng.random() < self.cfg.food_move_prob:
                dx, dy = DIRS4[int(self.rng.integers(0, 4))]
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny, self.cfg.grid_w, self.cfg.grid_h) and self.obstacles[ny, nx] == 0:
                    # avoid moving onto nests (keeps it clearer for gameplay)
                    if (nx, ny) not in {col.nest for col in self.colonies}:
                        moved[(nx, ny)] = moved.get((nx, ny), 0) + amt
                    else:
                        moved[(x, y)] = moved.get((x, y), 0) + amt
                else:
                    moved[(x, y)] = moved.get((x, y), 0) + amt
            else:
                moved[(x, y)] = moved.get((x, y), 0) + amt
        self.foods = moved

    def _respawn_food(self) -> None:
        if len(self.foods) >= self.cfg.max_food_sources:
            return
        if self.rng.random() < self.cfg.food_respawn_prob:
            forbidden = {col.nest for col in self.colonies} | set(self.foods.keys())
            x, y = random_empty_cell(self.rng, self.obstacles, forbidden)
            self.foods[(x, y)] = int(self.cfg.food_amount_init)

    def _cleanup_food(self) -> None:
        empty = [pos for pos, amt in self.foods.items() if amt <= 0]
        for pos in empty:
            del self.foods[pos]

    # --------------------------
    # Pheromone dynamics
    # --------------------------
    def _update_pheromones(self) -> None:
        for c in range(self.cfg.num_colonies):
            self.pheromone[c] = diffuse4(self.pheromone[c], self.obstacles, self.cfg.diff_rate)
            self.pheromone[c] = evaporate(self.pheromone[c], self.cfg.evap_rate)

    # --------------------------
    # Heuristic policy (for bots / demo)
    # --------------------------
    def heuristic_action(self, ant_idx: int) -> int:
        """
        Simple heuristic:
        - If carrying: go toward own nest (Manhattan decreasing)
        - Else: follow own pheromone gradient + exploration noise
        Returns action in {0,1,2,3} for DIRS4.
        """
        ant = self.ants[ant_idx]
        x, y = ant.x, ant.y
        colony = self.colonies[ant.colony_id]
        nx, ny = colony.nest

        candidates: List[Tuple[int, int, int]] = []  # (action, tx, ty)
        for a, (dx, dy) in enumerate(DIRS4):
            tx, ty = x + dx, y + dy
            if not in_bounds(tx, ty, self.cfg.grid_w, self.cfg.grid_h):
                continue
            if self.obstacles[ty, tx] == 1:
                continue
            candidates.append((a, tx, ty))

        if not candidates:
            return 0

        scores: List[float] = []
        if ant.carrying:
            d0 = abs(x - nx) + abs(y - ny)
            for a, tx, ty in candidates:
                d1 = abs(tx - nx) + abs(ty - ny)
                s = float(d0 - d1) * 2.0
                s += float(self.pheromone[ant.colony_id, ty, tx]) * 0.05
                s += float(self.rng.random()) * 0.1
                scores.append(s)
        else:
            for a, tx, ty in candidates:
                # follow pheromone but encourage exploration away from nest
                pher = float(self.pheromone[ant.colony_id, ty, tx])
                dist = abs(tx - nx) + abs(ty - ny)
                s = pher * 1.0 + dist * 0.02 + float(self.rng.random()) * 0.2
                scores.append(s)

        # sample via softmax-like
        m = max(scores)
        exps = [np.exp(s - m) for s in scores]
        total = float(np.sum(exps))
        r = float(self.rng.random()) * total
        acc = 0.0
        for (a, _, _), e in zip(candidates, exps):
            acc += float(e)
            if acc >= r:
                return int(a)
        return int(candidates[-1][0])

    # --------------------------
    # One tick update (key API)
    # --------------------------
    def step_tick(self, actions: Dict[int, int]) -> Tuple[Dict[int, float], Dict]:
        """
        Advance the world by ONE tick.
        - All ants deposit pheromone at their current position.
        - Only ants in `actions` will move this tick (others stay).
        - After movement & interactions, pheromones + food dynamics are updated once.

        Returns:
            rewards: dict[ant_idx] -> reward
            info: some world stats
        """
        rewards: Dict[int, float] = {}

        # 1) deposit for all ants
        for ant in self.ants:
            dep = self.cfg.deposit_return if ant.carrying else self.cfg.deposit_search
            self.pheromone[ant.colony_id, ant.y, ant.x] += float(dep)

        # 2) move selected ants + interactions
        # (we compute reward for those ants only)
        for ant_idx, act in actions.items():
            r = self._step_single_ant(ant_idx, act)
            rewards[ant_idx] = r

        # 3) update world fields
        self._update_pheromones()
        self._move_foods()
        self._respawn_food()
        self._cleanup_food()

        # 4) tick
        self.tick += 1

        info = {
            "tick": self.tick,
            "foods": len(self.foods),
            "delivered": [c.delivered for c in self.colonies],
        }
        return rewards, info

    def step_all_heuristic(self) -> Tuple[Dict[int, float], Dict]:
        """Convenience for demo: all ants move every tick using heuristic."""
        actions = {i: self.heuristic_action(i) for i in range(self.num_ants)}
        return self.step_tick(actions)

    # --------------------------
    # Per-ant transition + reward shaping
    # --------------------------
    def _step_single_ant(self, ant_idx: int, action: int) -> float:
        ant = self.ants[ant_idx]
        colony = self.colonies[ant.colony_id]
        x0, y0 = ant.x, ant.y

        # Step penalty
        reward = -0.01

        # Potential shaping target:
        # - If carrying: nest
        # - Else: nearest food
        if ant.carrying:
            target = colony.nest
            d0 = manhattan((x0, y0), target)
        else:
            d0 = self._distance_to_nearest_food((x0, y0))

        # Move attempt
        action = int(action) % 4
        dx, dy = DIRS4[action]
        x1, y1 = x0 + dx, y0 + dy

        collision = False
        if not in_bounds(x1, y1, self.cfg.grid_w, self.cfg.grid_h) or self.obstacles[y1, x1] == 1:
            # invalid move -> stay
            x1, y1 = x0, y0
            collision = True

        if collision:
            reward -= 0.05

        # Update occupancy counts if moved
        if (x1, y1) != (x0, y0):
            self.ant_counts[ant.colony_id, y0, x0] -= 1
            self.ant_counts[ant.colony_id, y1, x1] += 1
            ant.x, ant.y = x1, y1

        # Nest history update
        ant.steps_since_nest += 1

        # Pick up food
        if (not ant.carrying) and (ant.x, ant.y) in self.foods and self.foods[(ant.x, ant.y)] > 0:
            self.foods[(ant.x, ant.y)] -= 1
            ant.carrying = True
            reward += 1.0

        # Deliver food
        if ant.carrying and (ant.x, ant.y) == colony.nest:
            ant.carrying = False
            colony.delivered += 1
            ant.steps_since_nest = 0
            ant.nest_hits += 1
            reward += 5.0

        # Potential shaping
        if ant.carrying:
            d1 = manhattan((ant.x, ant.y), colony.nest)
        else:
            d1 = self._distance_to_nearest_food((ant.x, ant.y))

        if np.isfinite(d0) and np.isfinite(d1):
            reward += 0.02 * float(d0 - d1)

        return float(reward)

    def _distance_to_nearest_food(self, pos: Tuple[int, int]) -> float:
        if not self.foods:
            return float("inf")
        x, y = pos
        best = float("inf")
        for (fx, fy), amt in self.foods.items():
            if amt <= 0:
                continue
            d = abs(x - fx) + abs(y - fy)
            if d < best:
                best = d
        return best