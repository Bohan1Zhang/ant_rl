from __future__ import annotations

import argparse
from stable_baselines3 import PPO

from ant_sim.sim.entities import WorldConfig
from ant_sim.envs.ant_env_gym import AntColonyGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_ant_colony.zip")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = WorldConfig()
    env = AntColonyGymEnv(config=cfg, seed=args.seed)

    model = PPO.load(args.model)

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        if terminated or truncated:
            obs, info = env.reset(seed=args.seed)

    print("Eval done.")
    print("Total reward:", total_reward)
    print("Delivered:", info.get("delivered"))


if __name__ == "__main__":
    main()