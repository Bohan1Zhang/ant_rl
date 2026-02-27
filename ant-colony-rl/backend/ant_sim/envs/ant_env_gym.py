from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym

from ant_sim.sim.entities import WorldConfig
from ant_sim.sim.world import AntColonyWorld
from ant_sim.envs.obs import build_observation, make_observation_space


class AntColonyGymEnv(gym.Env):
    """
    Gymnasium env for "parameter sharing" training:
    - We keep ONE shared policy.
    - Each env step controls ONE "active ant" (round-robin across all ants and colonies).
    - A number of other ants also move using heuristic actions each env step to keep competition dynamics.
    - Observation is 5x5 multi-channel grid + stats (nest history included).
    - Action is 4-direction (Discrete(4)).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[WorldConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or WorldConfig()
        self.world = AntColonyWorld(self.cfg, seed=seed)

        # Observation: Dict(grid, stats)
        self.num_channels = 8
        self.stats_dim = 7
        self.observation_space = make_observation_space(self.num_channels, self.cfg.view_size, self.stats_dim)
        self.action_space = gym.spaces.Discrete(4)

        self.rng = np.random.default_rng(seed)
        self.active_ant: int = 0
        self.episode_steps: int = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.world.reset(seed=seed)
        self.active_ant = 0
        self.episode_steps = 0

        obs = build_observation(self.world, self.active_ant)
        info = {"tick": self.world.tick, "active_ant": self.active_ant, "delivered": [c.delivered for c in self.world.colonies]}
        return obs, info

    def step(self, action: int):
        self.episode_steps += 1

        # Choose which ants move this tick:
        # - active ant uses the RL action
        # - k other ants move using heuristic to create multi-colony competition dynamics
        actions: Dict[int, int] = {self.active_ant: int(action) % 4}

        k = int(min(self.cfg.bot_steps_per_env_step, self.world.num_ants - 1))
        if k > 0:
            # sample k other unique ants
            candidates = np.arange(self.world.num_ants)
            # ensure active ant excluded
            mask = candidates != self.active_ant
            others = candidates[mask]
            if len(others) > 0:
                pick = self.rng.choice(others, size=min(k, len(others)), replace=False)
                for idx in pick:
                    actions[int(idx)] = self.world.heuristic_action(int(idx))

        rewards, info = self.world.step_tick(actions)

        reward = float(rewards.get(self.active_ant, 0.0))

        # Round-robin parameter sharing
        self.active_ant = (self.active_ant + 1) % self.world.num_ants
        obs = build_observation(self.world, self.active_ant)

        terminated = False
        truncated = self.episode_steps >= self.cfg.max_episode_ticks

        info = {
            **info,
            "active_ant": self.active_ant,
        }
        return obs, reward, terminated, truncated, info