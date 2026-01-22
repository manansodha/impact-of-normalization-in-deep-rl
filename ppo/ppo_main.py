import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ale_py  # Registers Atari envs with Gymnasium
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ppo_helpers import Agent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TRAINING_METHODS: Tuple[str, ...] = (
    "vanilla",
    "reward_clip",
    "rbs",
    "grad_clip",
    "obs_norm",
    "adv_norm",
    "return_norm",
    "reward_norm",
)


@dataclass
class TrainConfig:
    env_id: str
    method: str
    render: bool
    batches: int = 1000
    episodes_per_batch: int = 5
    seeds: Tuple[int, ...] = (10, 20, 30,)
    clip_window: int = 2


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument(
        "--method",
        type=str,
        choices=TRAINING_METHODS,
        default="vanilla",
        help="PPO update method",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Pacman-v5",
        help="Gym environment name (e.g., ALE/Pacman-v5)",
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--clip_window",
        type=int,
        default=2,
        help="Batches used to refresh reward clip window",
    )
    args = parser.parse_args()
    return TrainConfig(
        env_id=args.env,
        method=args.method,
        render=args.render,
        clip_window=args.clip_window,
    )


def preprocess(obs: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=0).astype(np.float32) / 255.0


def make_env(env_id: str, render: bool) -> gym.Env:
    if render:
        return gym.make(env_id, render_mode="human")
    return gym.make(env_id)


def rollout_batch(
    env: gym.Env,
    agent: Agent,
    state: np.ndarray,
    episodes_per_batch: int,
    clip_bounds: Optional[Tuple[float, float]],
) -> Tuple[List[float], np.ndarray, int]:
    batch_returns: List[float] = []
    episodes_finished = 0

    for _ in range(episodes_per_batch):
        ep_rewards: List[float] = []
        done = False

        while not done:
            action, logp, value = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)

            ep_rewards.append(reward)
            agent.remember(state, action, reward, done, logp, value, next_state)
            state = next_state

        ep_return = float(sum(ep_rewards))
        if clip_bounds is not None:
            low, high = clip_bounds
            ep_return = float(np.clip(ep_return, low, high))

        batch_returns.append(ep_return)
        episodes_finished += 1

        obs, _ = env.reset()
        state = preprocess(obs)

    return batch_returns, state, episodes_finished


def add_seed_stats(frames: List[pd.DataFrame], seeds: Sequence[int]) -> List[pd.DataFrame]:
    enriched: List[pd.DataFrame] = []
    seed_cols = list(seeds)
    for df in frames:
        seed_data = df[seed_cols]
        df["Avg"] = seed_data.mean(axis=1)
        df["Std"] = seed_data.std(axis=1)
        df["High"] = seed_data.max(axis=1)
        df["Low"] = seed_data.min(axis=1)
        enriched.append(df)
    return enriched


def choose_update(agent: Agent, method: str) -> float:
    if method == "vanilla":
        return agent.vanilla_ppo_update()
    if method == "grad_clip":
        return agent.update_gradient_clipping()
    if method == "obs_norm":
        return agent.update_obs_norm()
    if method == "adv_norm":
        return agent.update_adv_norm()
    if method == "return_norm":
        return agent.update_return_norm()
    if method == "reward_norm":
        return agent.update_reward_norm()
    if method == "rbs":
        return agent.update_rbs()
    raise ValueError(f"Unknown method: {method}")


def collect_loss_deltas(agent: Agent, prev_policy: int, prev_value: int) -> Tuple[float, float]:
    policy_slice = agent.policy_loss_history[prev_policy:]
    value_slice = agent.value_loss_history[prev_value:]

    policy_loss = float(np.mean(policy_slice)) if policy_slice else float("nan")
    value_loss = float(np.mean(value_slice)) if value_slice else float("nan")
    return policy_loss, value_loss


def plot_results(
    loss_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    value_df: pd.DataFrame,
    method: str,
) -> None:
    fig = plt.figure(figsize=(15, 10))

    ax2 = plt.subplot(221)
    ax2.fill_between(loss_df.index, loss_df["Low"], loss_df["High"], color="#A8DADC", alpha=0.5, label="High-Low")
    ax2.plot(loss_df["Avg"], label="Avg Loss", color="#1D3557", linewidth=2)
    ax2.set_ylabel("Average PPO Loss")
    ax2.set_xlabel("PPO Update")
    ax2.legend()

    ax3 = plt.subplot(222)
    ax3.fill_between(reward_df.index, reward_df["Low"], reward_df["High"], color="#FEDCC8", alpha=0.5, label="High-Low")
    ax3.plot(reward_df["Avg"], label="Avg Reward", color="#E63946", linewidth=2)
    ax3.set_ylabel("Average Reward")
    ax3.set_xlabel("PPO Update")
    ax3.legend()

    ax4 = plt.subplot(223)
    ax4.fill_between(policy_df.index, policy_df["Low"], policy_df["High"], color="#B0E0A0", alpha=0.5, label="High-Low")
    ax4.plot(policy_df["Avg"], label="Policy Loss", color="#38B000", linewidth=2)
    ax4.set_ylabel("Average Policy Loss")
    ax4.set_xlabel("PPO Update")
    ax4.legend()

    ax5 = plt.subplot(224)
    ax5.fill_between(value_df.index, value_df["Low"], value_df["High"], color="#D7BDE2", alpha=0.5, label="High-Low")
    ax5.plot(value_df["Avg"], label="Value Loss", color="#8E44AD", linewidth=2)
    ax5.set_ylabel("Average Value Loss")
    ax5.set_xlabel("PPO Update")
    ax5.legend()

    fig.suptitle(f"PPO Training Stability - {method}", fontsize=16, fontweight="bold")
    plt.show()


def train_seed(env: gym.Env, dummy_obs_space: Box, config: TrainConfig, seed: int) -> Dict[str, List[float]]:
    agent = Agent(
        obs_space=dummy_obs_space,
        action_space=env.action_space,
        hidden=64,
        lr=2.5e-2,
        gamma=0.997,
        clip_coef=0.1,
        entropy_coef=0.0001,
        value_coef=0.5,
        seed=seed,
        batch_size=64,
        ppo_epochs=10,
        lam=0.95,
    )

    obs, _ = env.reset(seed=seed)
    state = preprocess(obs)

    clip_bounds: Optional[Tuple[float, float]] = None
    alpha = np.random.uniform(1, 2) if config.method == "reward_clip" else None
    if config.method == "reward_clip":
        logger.info("Clip Coefficient= %f", alpha)

    loss_history: List[float] = []
    reward_history: List[float] = []
    policy_loss_history: List[float] = []
    value_loss_history: List[float] = []

    episode = 0
    total_return = 0.0

    for update in range(1, config.batches + 1):
        prev_policy_len = len(agent.policy_loss_history)
        prev_value_len = len(agent.value_loss_history)

        batch_returns, state, ep_finished = rollout_batch(
            env=env,
            agent=agent,
            state=state,
            episodes_per_batch=config.episodes_per_batch,
            clip_bounds=clip_bounds,
        )

        episode += ep_finished
        total_return += float(sum(batch_returns))

        if config.method == "reward_clip":
            assert alpha is not None
            mu = float(np.mean(batch_returns))
            sigma = float(np.std(batch_returns) + 1e-8)
            clip_bounds = (mu - alpha * sigma, mu + alpha * sigma)
            avg_loss = agent.vanilla_ppo_update()
            reward_history.append(mu)
            logger.info(
                "Update %d: batch_mean=%.4f, batch_std=%.4f, clip_range=[%.3f, %.3f], episodes=%d",
                update,
                mu,
                sigma,
                clip_bounds[0],
                clip_bounds[1],
                episode,
            )
        else:
            avg_loss = choose_update(agent, config.method)
            avg_ret = total_return / max(episode, 1)
            reward_history.append(avg_ret)
            logger.info(
                "Update %d: episodes=%d, avg_return=%.2f, avg_loss=%.4f",
                update,
                episode,
                avg_ret,
                avg_loss,
            )

        policy_loss, value_loss = collect_loss_deltas(agent, prev_policy_len, prev_value_len)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        loss_history.append(avg_loss)

    return {
        "loss": loss_history,
        "reward": reward_history,
        "policy_loss": policy_loss_history,
        "value_loss": value_loss_history,
        "adv_mean": agent.adv_mean,
        "adv_std": agent.adv_std,
    }


def main() -> int:
    config = parse_args()

    env = make_env(config.env_id, config.render)
    logger.info("Observation space: %s", env.observation_space)
    logger.info("Action space: %s", env.action_space)
    logger.info("Method: %s", config.method)

    obs, _ = env.reset()
    dummy_obs_space = Box(low=0.0, high=1.0, shape=preprocess(obs).shape)

    seed_results: Dict[int, Dict[str, List[float]]] = {}

    try:
        for seed in config.seeds:
            seed_results[seed] = train_seed(env, dummy_obs_space, config, seed)

        loss_df = pd.DataFrame(
            {seed: pd.Series(res["loss"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )
        reward_df = pd.DataFrame(
            {seed: pd.Series(res["reward"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )
        policy_df = pd.DataFrame(
            {seed: pd.Series(res["policy_loss"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )
        value_df = pd.DataFrame(
            {seed: pd.Series(res["value_loss"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )
        adv_mean_df = pd.DataFrame(
            {seed: pd.Series(res["adv_mean"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )
        adv_std_df = pd.DataFrame(
            {seed: pd.Series(res["adv_std"]).reindex(range(1, config.batches + 1)) for seed, res in seed_results.items()}
        )

        loss_df, reward_df, policy_df, value_df = add_seed_stats(
            [loss_df, reward_df, policy_df, value_df], config.seeds
        )
        adv_mean_df, adv_std_df = add_seed_stats([adv_mean_df, adv_std_df], config.seeds)

        loss_df.to_csv(f"{config.method}_loss_history.csv")
        reward_df.to_csv(f"{config.method}_reward_history.csv")
        policy_df.to_csv(f"{config.method}_policy_loss.csv")
        value_df.to_csv(f"{config.method}_value_loss.csv")
        adv_mean_df.to_csv(f"{config.method}_adv_mean.csv")
        adv_std_df.to_csv(f"{config.method}_adv_std.csv")

        plot_results(loss_df, reward_df, policy_df, value_df, config.method)

    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)
        return 1
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
