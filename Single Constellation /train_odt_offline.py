import os
import pickle
import random

import numpy as np
import torch

from ODT import OnlineDecisionTransformer


def _get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "odt_offline_dataset.pkl")
    output_path = os.path.join(base_dir, "decision_transformer_offline.pth")

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

    model = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=10,
        embed_dim=32,
        num_layers=1,
        learning_rate=1e-6,
        target_return=1.0,
        buffer_size=len(trajectories),
        device=device,
    )
    model.optimizer.param_groups[0]["weight_decay"] = 1e-2

    for traj in trajectories:
        model.buffer.add_trajectory(traj)

    epochs = 10000
    steps_per_epoch = 80
    batch_size = 32

    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
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
            loss = torch.nn.functional.cross_entropy(
                action_logits[:, :-1].reshape(-1, action_dim),
                target_actions.reshape(-1)
            )

            if not torch.isfinite(loss):
                print("Skipping update with non-finite loss.")
                continue
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=0.5)
            model.optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.6f}")

    model.save(output_path)
    print(f"Saved offline ODT model to {output_path}")


if __name__ == "__main__":
    main()
