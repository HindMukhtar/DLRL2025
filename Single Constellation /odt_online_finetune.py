import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sb3_contrib.common.wrappers import ActionMasker

from LEOEnvironmentRL import load_route_from_csv
from ODT import ExperienceBuffer, LEOEnv, OnlineDecisionTransformer, predict_valid_action_dt


def mask_fn(env):
    return env.action_mask


def _build_agent_from_checkpoint(model_path: str, buffer_size: int = 2000):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dim = checkpoint["model_state_dict"]["state_embedding.weight"].shape[1]
    action_dim = checkpoint["model_state_dict"]["action_embedding.weight"].shape[0]
    embed_dim = checkpoint["model_state_dict"]["state_embedding.weight"].shape[0]
    pos_len = checkpoint["model_state_dict"]["pos_embedding"].shape[1]
    max_length = pos_len // 3

    block_indices = []
    for key in checkpoint["model_state_dict"].keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    num_layers = max(block_indices) + 1 if block_indices else 1
    target_return = float(checkpoint.get("target_return", 1.0))

    agent = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=max_length,
        embed_dim=embed_dim,
        num_layers=num_layers,
        learning_rate=3e-6,
        target_return=target_return,
        buffer_size=buffer_size,
        device="cpu",
    )
    agent.load(model_path)
    agent.set_eval_mode(False)
    return agent, state_dim, action_dim


def _load_offline_buffer(dataset_path: str, max_trajs: int = 1200):
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    if not trajectories:
        raise RuntimeError("Offline dataset is empty.")

    # Keep a deterministic subset for memory/perf stability.
    if len(trajectories) > max_trajs:
        trajectories = trajectories[:max_trajs]

    offline_buffer = ExperienceBuffer(max_size=len(trajectories))
    for traj in trajectories:
        offline_buffer.add_trajectory(traj)
    return offline_buffer, trajectories


def _compute_rtg_candidates(trajectories, percentiles=(50, 75, 90), gamma=0.99):
    all_rewards = np.concatenate([traj["rewards"] for traj in trajectories])
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards)) if np.std(all_rewards) > 0 else 1.0

    returns = []
    for traj in trajectories:
        rewards = (np.asarray(traj["rewards"], dtype=np.float32) - reward_mean) / reward_std
        rewards = np.clip(rewards, -5.0, 5.0)
        ret = 0.0
        for r in rewards[::-1]:
            ret = float(r) + gamma * ret
        returns.append(ret)

    cands = [float(np.clip(np.percentile(returns, p), -10.0, 10.0)) for p in percentiles]
    return cands


def _concat_batches(batches):
    out = {}
    for key in batches[0].keys():
        out[key] = torch.cat([b[key] for b in batches if b[key].numel() > 0], dim=0)
    return out


def _train_step_mixed(agent, offline_buffer, batch_size=64, offline_frac=0.7):
    if len(offline_buffer.trajectories) == 0 or len(agent.buffer.trajectories) == 0:
        return None

    offline_bs = max(1, int(round(batch_size * offline_frac)))
    online_bs = max(1, batch_size - offline_bs)

    b_off = offline_buffer.sample_batch(offline_bs, agent.max_length)
    b_on = agent.buffer.sample_batch(online_bs, agent.max_length)
    batch = _concat_batches([b_off, b_on])

    for key in batch:
        batch[key] = batch[key].to(agent.device)

    agent.model.train()
    outputs = agent.model(
        batch["returns_to_go"],
        batch["states"],
        batch["actions"],
        batch["timesteps"],
    )
    action_logits = outputs["action_logits"]
    target_actions = batch["actions"][:, 1:]

    if action_logits.shape[1] == 0 or target_actions.shape[1] == 0:
        return None

    min_len = min(action_logits.shape[1], target_actions.shape[1])
    action_logits = action_logits[:, :min_len]
    target_actions = target_actions[:, :min_len]

    loss = F.cross_entropy(
        action_logits.reshape(-1, action_logits.shape[-1]),
        target_actions.reshape(-1),
        reduction="mean",
    )

    if not torch.isfinite(loss):
        return None

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=0.5)
    agent.optimizer.step()
    return float(loss.item())


def _evaluate_target_return(agent, constellation_name, route, state_dim, action_dim, target_return, scenarios):
    old_target = float(agent.target_return)
    agent.target_return = float(target_return)
    agent.set_eval_mode(True)

    total_rewards = []
    for scenario in scenarios:
        env = LEOEnv(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            mask = env.env._get_action_mask()[:action_dim]
            action = predict_valid_action_dt(agent, obs[:state_dim], mask)
            next_obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            obs = next_obs
        total_rewards.append(ep_reward)

    score = float(np.mean(total_rewards)) if total_rewards else -1e9
    agent.target_return = old_target
    agent.set_eval_mode(False)
    return score


def _evaluate_composite(agent, constellation_name, route, state_dim, action_dim, scenarios):
    agent.set_eval_mode(True)
    rows = []
    for scenario in scenarios:
        env = LEOEnv(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        obs, info = env.reset()
        done = False
        truncated = False
        rewards = 0.0
        ratios = []
        drops = []
        handovers = []
        latency_viol = []

        while not (done or truncated):
            mask = env.env._get_action_mask()[:action_dim]
            action = predict_valid_action_dt(agent, obs[:state_dim], mask)
            next_obs, reward, done, truncated, info = env.step(action)
            rewards += float(reward)
            ratios.append(float(next_obs[7]))
            drops.append(float(next_obs[15]))
            handovers.append(float(next_obs[5]))
            latency_s = float(next_obs[10]) + float(next_obs[11])
            latency_req = float(next_obs[13]) if float(next_obs[13]) > 0 else 1e9
            latency_viol.append(1.0 if latency_s > latency_req else 0.0)
            obs = next_obs

        rows.append(
            {
                "reward": rewards,
                "ratio": float(np.mean(ratios)) if ratios else 0.0,
                "drop": float(np.sum(drops)) if drops else 0.0,
                "handover": float(max(handovers)) if handovers else 0.0,
                "lat_viol": float(np.mean(latency_viol)) if latency_viol else 0.0,
            }
        )

    agent.set_eval_mode(False)
    if not rows:
        return -1e9
    df = pd.DataFrame(rows)
    return float(
        (1.0 * df["ratio"].mean())
        - (0.6 * df["lat_viol"].mean())
        - (0.02 * df["drop"].mean())
        - (0.001 * df["handover"].mean())
    )


def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "decision_transformer_offline.pth")
    dataset_path = os.path.join(base_dir, "odt_offline_dataset.pkl")
    output_path = os.path.join(base_dir, "decision_transformer_online_finetune.pth")
    best_output_path = os.path.join(base_dir, "decision_transformer_online_finetune_best.pth")

    input_params = pd.read_csv(os.path.join(base_dir, "input.csv"))
    constellation_name = input_params["Constellation"][0]
    route, _ = load_route_from_csv(os.path.join(base_dir, "route_5s_interpolated.csv"), skip_rows=0)

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Weighted scenario schedule (focus weak scenarios).
    scenario_weights = {
        None: 3,
        "load_cycle_1": 3,
        "load_cycle_2": 1,
        "load_cycle_5": 1,
        "medium_aircraft": 1,
        "large_aircraft": 1,
        "snr_congested": 1,
    }
    sweep_scenarios = [None, "load_cycle_1"]
    eval_scenarios = [None, "load_cycle_1", "large_aircraft", "snr_congested"]

    finetune_rounds = 3
    train_interval = 5
    batch_size = 96
    offline_frac = 0.7
    rtg_sweep_every_episodes = 4
    eval_every_episodes = 2

    agent, state_dim, action_dim = _build_agent_from_checkpoint(model_path, buffer_size=2000)
    # Smaller LR for stable fine-tuning.
    agent.optimizer.param_groups[0]["lr"] = 3e-6
    agent.optimizer.param_groups[0]["weight_decay"] = 1e-2

    offline_buffer, trajectories = _load_offline_buffer(dataset_path, max_trajs=1200)
    rtg_candidates = _compute_rtg_candidates(trajectories, percentiles=(50, 75, 90))
    print(f"RTG candidates (p50/p75/p90): {rtg_candidates}")

    # Warm start online buffer with a subset of offline trajectories.
    warm_online = min(300, len(offline_buffer.trajectories))
    for i in range(warm_online):
        agent.buffer.add_trajectory(offline_buffer.trajectories[i])

    # Initial RTG sweep.
    best_rtg = agent.target_return
    best_rtg_score = -1e9
    for cand in rtg_candidates:
        score = _evaluate_target_return(
            agent, constellation_name, route, state_dim, action_dim, cand, sweep_scenarios
        )
        print(f"[RTG sweep init] target={cand:.4f}, score={score:.4f}")
        if score > best_rtg_score:
            best_rtg_score = score
            best_rtg = cand
    agent.target_return = float(best_rtg)
    print(f"Selected initial RTG target: {agent.target_return:.4f}")

    # Build weighted schedule.
    schedule = []
    for _ in range(finetune_rounds):
        round_scenarios = []
        for s, w in scenario_weights.items():
            round_scenarios.extend([s] * int(w))
        random.shuffle(round_scenarios)
        schedule.extend(round_scenarios)

    best_eval = -1e9
    global_step = 0
    losses = []

    for ep_idx, scenario in enumerate(schedule, start=1):
        print(f"Fine-tuning episode {ep_idx}/{len(schedule)} on scenario: {scenario if scenario else 'no_scenario'}")
        env = LEOEnv(constellation_name, route, scenario=scenario)
        env = ActionMasker(env, mask_fn)
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            mask = env.env._get_action_mask()[:action_dim]
            action = predict_valid_action_dt(agent, obs[:state_dim], mask)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.step(obs[:state_dim], action, reward, next_obs[:state_dim], done or truncated)

            if global_step % train_interval == 0:
                loss_val = _train_step_mixed(
                    agent,
                    offline_buffer,
                    batch_size=batch_size,
                    offline_frac=offline_frac,
                )
                if loss_val is not None:
                    losses.append(loss_val)

            obs = next_obs
            global_step += 1

        if ep_idx % rtg_sweep_every_episodes == 0:
            best_rtg = agent.target_return
            best_rtg_score = -1e9
            for cand in rtg_candidates:
                score = _evaluate_target_return(
                    agent, constellation_name, route, state_dim, action_dim, cand, sweep_scenarios
                )
                if score > best_rtg_score:
                    best_rtg_score = score
                    best_rtg = cand
            agent.target_return = float(best_rtg)
            print(f"[RTG sweep] updated target_return={agent.target_return:.4f}, score={best_rtg_score:.4f}")

        if ep_idx % eval_every_episodes == 0:
            eval_score = _evaluate_composite(
                agent, constellation_name, route, state_dim, action_dim, eval_scenarios
            )
            mean_loss = float(np.mean(losses[-200:])) if losses else float("nan")
            print(f"[Eval] ep={ep_idx}, composite={eval_score:.4f}, mean_recent_loss={mean_loss:.6f}")
            if eval_score > best_eval:
                best_eval = eval_score
                agent.save(best_output_path)
                print(f"[Eval] New best checkpoint saved: {best_output_path}")

    agent.save(output_path)
    print(f"Saved fine-tuned ODT model to {output_path}")
    print(f"Best eval composite: {best_eval:.4f}")


if __name__ == "__main__":
    main()
