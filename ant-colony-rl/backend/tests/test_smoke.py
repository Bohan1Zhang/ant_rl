from __future__ import annotations

import sys
import os

THIS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ant_sim.sim.entities import WorldConfig
from ant_sim.envs.ant_env_gym import AntColonyGymEnv


def main():
    cfg = WorldConfig(
        grid_w=30,
        grid_h=20,
        num_colonies=2,
        ants_per_colony=10,
        num_food_init=3,
        max_episode_ticks=200,
        bot_steps_per_env_step=5,
    )
    env = AntColonyGymEnv(config=cfg, seed=0)
    obs, info = env.reset(seed=0)

    assert "grid" in obs and "stats" in obs
    assert obs["grid"].shape == (8, 5, 5)
    assert obs["stats"].shape == (7,)

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs["grid"].shape == (8, 5, 5)
        assert obs["stats"].shape == (7,)
        if truncated or terminated:
            obs, info = env.reset()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()