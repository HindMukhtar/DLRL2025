import os
import random

import numpy as np
import pandas as pd
import torch
from sb3_contrib.common.wrappers import ActionMasker

from HandoverEnvironment_ODT import LEOEnv, predict_valid_action_dt
from LEOEnvironmentRL import load_route_from_csv
from ODT import OnlineDecisionTransformer


def mask_fn(env):
    return env.action_mask


def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "decision_transformer_offline.pth")
    output_path = os.path.join(base_dir, "decision_transformer_online_finetune.pth")

    input_params = pd.read_csv(os.path.join(base_dir, "input.csv"))
    constellation_name = input_params["Constellation"][0]
    route, route_duration = load_route_from_csv(
        os.path.join(base_dir, "route_5s_interpolated.csv"), skip_rows=0
    )

    env = LEOEnv(constellation_name, route, scenario=None)
    env = ActionMasker(env, mask_fn)

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dim = checkpoint["model_state_dict"]["state_embedding.weight"].shape[1]
    action_dim = checkpoint["model_state_dict"]["action_embedding.weight"].shape[0]

    agent = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=10,
        embed_dim=32,
        num_layers=1,
        target_return=1.0,
        buffer_size=200,
    )
    agent.load(model_path)
    agent.set_eval_mode(False)

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0
    train_interval = 10
    batch_size = 32

    while not (done or truncated):
        mask = env.env._get_action_mask()[:action_dim]
        action = predict_valid_action_dt(agent, obs[:state_dim], mask)

        next_obs, reward, done, truncated, info = env.step(action)
        agent.step(obs[:state_dim], action, reward, next_obs[:state_dim], done or truncated)

        if step_count % train_interval == 0:
            agent.train_step(batch_size=batch_size)

        obs = next_obs
        step_count += 1

    agent.save(output_path)
    print(f"Saved fine-tuned ODT model to {output_path}")


if __name__ == "__main__":
    main()
