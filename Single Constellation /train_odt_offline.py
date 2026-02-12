import os
import pickle
import random

import numpy as np
import torch

from ODT import ExperienceBuffer, OnlineDecisionTransformer


def _get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _scenario_name(traj):
    scenario = traj.get("scenario", None)
    if scenario is None:
        return "no_scenario"
    return str(scenario)


def _split_easy_hard(trajectories):
    """
    Split trajectories into easy/hard regimes using scenario metadata if present.
    Fallback: if metadata is unavailable, return empty split to trigger uniform sampling.
    """
    # Regime definition can be tuned.
    easy = {"load_cycle_1", "load_cycle_2"}
    hard = {"load_cycle_5", "large_aircraft", "snr_congested", "medium_aircraft"}

    has_scenario_meta = any("scenario" in t for t in trajectories)
    if not has_scenario_meta:
        return [], []

    easy_trajs = []
    hard_trajs = []
    for traj in trajectories:
        s = _scenario_name(traj)
        if s in easy:
            easy_trajs.append(traj)
        elif s in hard:
            hard_trajs.append(traj)

    return easy_trajs, hard_trajs


def _concat_batches(batches):
    out = {}
    for key in batches[0].keys():
        tensors = [b[key] for b in batches if key in b and b[key].numel() > 0]
        out[key] = torch.cat(tensors, dim=0) if tensors else torch.empty(0)
    return out


def main():
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "odt_offline_dataset.pkl")
    output_path = os.path.join(base_dir, "decision_transformer_offline.pth")
    best_path = os.path.join(base_dir, "decision_transformer_offline_best.pth")
    #resume_path = best_path if os.path.exists(best_path) else None

    resume_path = None 
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    if not trajectories:
        raise RuntimeError("No trajectories found in offline dataset.")

    all_rewards = np.concatenate([traj["rewards"] for traj in trajectories])
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards)) if np.std(all_rewards) > 0 else 1.0

    for traj in trajectories:
        rewards = (traj["rewards"] - reward_mean) / reward_std
        traj["rewards"] = np.clip(rewards, -5.0, 5.0)

    state_dim = trajectories[0]["states"].shape[1]
    max_action = max(int(np.max(traj["actions"])) for traj in trajectories)
    action_dim = max_action + 1

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    device = "cpu"
    print(f"Offline ODT device: {device}")

    # Calibrate RTG target from normalized discounted trajectory returns (75th percentile).
    gamma = 0.99
    traj_returns = []
    for traj in trajectories:
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        if rewards.size == 0:
            continue
        ret = 0.0
        for r in rewards[::-1]:
            ret = float(r) + gamma * ret
        traj_returns.append(ret)
    calibrated_target_return = float(np.percentile(traj_returns, 75)) if traj_returns else 1.0
    calibrated_target_return = float(np.clip(calibrated_target_return, -10.0, 10.0))
    print(f"Calibrated RTG target (p75): {calibrated_target_return:.4f}")

    model = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=20,
        embed_dim=64,
        num_layers=2,
        learning_rate=1e-5,
        target_return=calibrated_target_return,
        buffer_size=len(trajectories),
        device=device,
    )
    model.optimizer.param_groups[0]["weight_decay"] = 1e-2

    # Resume training from last best checkpoint if available.
    if resume_path is not None:
        print(f"Resuming offline ODT from: {resume_path}")
        model.load(resume_path)

    for traj in trajectories:
        model.buffer.add_trajectory(traj)

    easy_trajs, hard_trajs = _split_easy_hard(trajectories)
    use_regime_balanced = len(easy_trajs) > 0 and len(hard_trajs) > 0
    if use_regime_balanced:
        easy_buffer = ExperienceBuffer(max_size=len(easy_trajs))
        hard_buffer = ExperienceBuffer(max_size=len(hard_trajs))
        for traj in easy_trajs:
            easy_buffer.add_trajectory(traj)
        for traj in hard_trajs:
            hard_buffer.add_trajectory(traj)
        print(
            f"Regime-balanced sampling enabled: easy={len(easy_trajs)} hard={len(hard_trajs)} (50/50 per batch)"
        )
    else:
        easy_buffer = None
        hard_buffer = None
        print("Regime metadata missing or incomplete; using uniform sampling from full buffer.")

    epochs = 1000
    steps_per_epoch = 200
    batch_size = 128

    best_loss = float("inf")
    for epoch in range(epochs):
        losses = []
        for step_idx in range(steps_per_epoch):
            if use_regime_balanced:
                easy_bs = batch_size // 2
                hard_bs = batch_size - easy_bs
                b_easy = easy_buffer.sample_batch(easy_bs, model.max_length)
                b_hard = hard_buffer.sample_batch(hard_bs, model.max_length)
                batch = _concat_batches([b_easy, b_hard])
            else:
                batch = model.buffer.sample_batch(batch_size, model.max_length)
            has_non_finite = any(
                (~torch.isfinite(batch[key])).any().item() for key in batch
            )
            if has_non_finite:
                print("Skipping batch with non-finite values.")
                continue

            for key in batch:
                batch[key] = batch[key].to(model.device)
            batch["returns_to_go"] = torch.clamp(batch["returns_to_go"], -10.0, 10.0)

            model.model.train()
            outputs = model.model(
                batch["returns_to_go"],
                batch["states"],
                batch["actions"],
                batch["timesteps"]
            )
            action_logits = outputs["action_logits"]
            target_actions = batch["actions"][:, 1:]
            if action_logits.shape[1] == 0 or target_actions.shape[1] == 0:
                continue
            min_len = min(action_logits.shape[1], target_actions.shape[1])
            loss = torch.nn.functional.cross_entropy(
                action_logits[:, :min_len].reshape(-1, action_dim),
                target_actions[:, :min_len].reshape(-1)
            )

            if not torch.isfinite(loss):
                print("Skipping update with non-finite loss.")
                continue
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=0.5)
            model.optimizer.step()
            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                step_checkpoint = os.path.join(
                    base_dir,
                    f"decision_transformer_offline_best.pth"
                )
                model.save(step_checkpoint)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.6f}")

    model.save(output_path)
    print(f"Saved offline ODT model to {output_path}")


if __name__ == "__main__":
    main()
