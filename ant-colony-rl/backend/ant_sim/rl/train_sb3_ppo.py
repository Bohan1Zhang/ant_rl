from __future__ import annotations

import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ant_sim.sim.entities import WorldConfig
from ant_sim.envs.ant_env_gym import AntColonyGymEnv


def make_env(seed: int, cfg: WorldConfig):
    def _thunk():
        env = AntColonyGymEnv(config=cfg, seed=seed)
        return Monitor(env)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-dir", type=str, default=os.path.join("models"))
    parser.add_argument("--tensorboard-dir", type=str, default=os.path.join("runs"))
    args = parser.parse_args()

    cfg = WorldConfig(
        # keep defaults; you can tweak here for curriculum later
        view_size=5,
        max_episode_ticks=3000,
        num_colonies=2,
        ants_per_colony=80,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.seed + i, cfg) for i in range(args.n_envs)])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.tensorboard_dir,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.total_timesteps)

    out_path = os.path.join(args.model_dir, "ppo_ant_colony")
    model.save(out_path)
    print(f"Saved model to: {out_path}.zip")


if __name__ == "__main__":
    main()