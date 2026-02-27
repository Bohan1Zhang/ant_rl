from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class WorldConfig:
    # Grid
    grid_w: int = 80
    grid_h: int = 60
    obstacle_ratio: float = 0.06

    # Colonies / ants
    num_colonies: int = 2
    ants_per_colony: int = 80

    # Food
    num_food_init: int = 6
    food_amount_init: int = 250
    max_food_sources: int = 12
    food_respawn_prob: float = 0.01  # per tick
    food_move_prob: float = 0.02     # per tick

    # Pheromone dynamics
    evap_rate: float = 0.02
    diff_rate: float = 0.15
    deposit_search: float = 0.8
    deposit_return: float = 2.5

    # RL / episode
    max_episode_ticks: int = 3000  # how many world ticks per episode (Gym truncation)
    view_size: int = 5

    # Observation normalization
    pheromone_clip: float = 10.0
    food_clip: float = 250.0
    ant_count_clip: float = 6.0

    # Bot stepping inside Gym env (competition dynamics)
    bot_steps_per_env_step: int = 20  # how many other ants (heuristic) move each Gym step


@dataclass
class Colony:
    colony_id: int
    nest: Tuple[int, int]
    delivered: int = 0


@dataclass
class Ant:
    ant_id: int
    colony_id: int
    x: int
    y: int
    carrying: bool = False

    # "nest history" (as you requested)
    steps_since_nest: int = 0
    nest_hits: int = 0