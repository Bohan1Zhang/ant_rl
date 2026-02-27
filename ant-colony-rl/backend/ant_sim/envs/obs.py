from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import gymnasium as gym

from ant_sim.sim.world import AntColonyWorld
from ant_sim.sim.utils import in_bounds


def make_observation_space(num_channels: int, view_size: int, stats_dim: int) -> gym.spaces.Dict:
    grid_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(num_channels, view_size, view_size),
        dtype=np.float32,
    )
    stats_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(stats_dim,),
        dtype=np.float32,
    )
    return gym.spaces.Dict({"grid": grid_space, "stats": stats_space})


def build_observation(world: AntColonyWorld, ant_idx: int) -> Dict[str, np.ndarray]:
    """
    5x5 local view with channels:
      0 obstacle_or_oob
      1 food_amount_norm
      2 own_nest
      3 other_nest
      4 own_pheromone_norm
      5 enemy_pheromone_max_norm
      6 ally_ant_density_norm
      7 enemy_ant_density_norm

    stats vector (nest history + useful extras):
      [0] carrying (0/1)
      [1] steps_since_nest_norm
      [2] nest_hits_norm
      [3] dx_to_own_nest_norm (-1..1)
      [4] dy_to_own_nest_norm (-1..1)
      [5] colony_id_norm (0..1)
      [6] time_norm (0..1)
    """
    cfg = world.cfg
    ant = world.ants[ant_idx]
    c = ant.colony_id
    H, W = cfg.grid_h, cfg.grid_w
    view = cfg.view_size
    half = view // 2

    C = 8
    grid = np.zeros((C, view, view), dtype=np.float32)

    # Pre-calc nests
    own_nest = world.colonies[c].nest
    other_nests = {col.nest for col in world.colonies if col.colony_id != c}

    for j in range(view):
        for i in range(view):
            gx = ant.x + (i - half)
            gy = ant.y + (j - half)

            if not in_bounds(gx, gy, W, H):
                grid[0, j, i] = 1.0  # obstacle/oob
                continue

            if world.obstacles[gy, gx] == 1:
                grid[0, j, i] = 1.0

            # food
            amt = world.foods.get((gx, gy), 0)
            if amt > 0:
                grid[1, j, i] = float(min(amt, cfg.food_clip)) / float(cfg.food_clip)

            # nests
            if (gx, gy) == own_nest:
                grid[2, j, i] = 1.0
            if (gx, gy) in other_nests:
                grid[3, j, i] = 1.0

            # pheromones
            own_ph = float(world.pheromone[c, gy, gx])
            grid[4, j, i] = float(min(own_ph, cfg.pheromone_clip)) / float(cfg.pheromone_clip)

            if cfg.num_colonies > 1:
                enemy_max = 0.0
                for cc in range(cfg.num_colonies):
                    if cc == c:
                        continue
                    enemy_max = max(enemy_max, float(world.pheromone[cc, gy, gx]))
                grid[5, j, i] = float(min(enemy_max, cfg.pheromone_clip)) / float(cfg.pheromone_clip)

            # ants density (ally / enemy)
            ally = int(world.ant_counts[c, gy, gx])
            enemy = 0
            for cc in range(cfg.num_colonies):
                if cc == c:
                    continue
                enemy += int(world.ant_counts[cc, gy, gx])

            grid[6, j, i] = float(min(ally, cfg.ant_count_clip)) / float(cfg.ant_count_clip)
            grid[7, j, i] = float(min(enemy, cfg.ant_count_clip)) / float(cfg.ant_count_clip)

    # stats
    nx, ny = own_nest
    dx = (nx - ant.x) / max(1.0, float(cfg.grid_w))
    dy = (ny - ant.y) / max(1.0, float(cfg.grid_h))
    colony_norm = 0.0 if cfg.num_colonies <= 1 else (c / float(cfg.num_colonies - 1))
    time_norm = float(min(world.tick, cfg.max_episode_ticks)) / float(cfg.max_episode_ticks)

    steps_norm = float(min(ant.steps_since_nest, 200)) / 200.0
    hits_norm = float(min(ant.nest_hits, 50)) / 50.0

    stats = np.array(
        [
            1.0 if ant.carrying else 0.0,
            steps_norm,
            hits_norm,
            float(np.clip(dx, -1.0, 1.0)),
            float(np.clip(dy, -1.0, 1.0)),
            float(np.clip(colony_norm, 0.0, 1.0)),
            float(np.clip(time_norm, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return {"grid": grid, "stats": stats}