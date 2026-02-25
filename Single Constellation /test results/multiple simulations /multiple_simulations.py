import argparse
import os
import subprocess
import sys


def _run_once(testscript_path, model, scenario, seed, run_tag):
    env = os.environ.copy()
    env["SELECTED_MODEL"] = model
    env["SCENARIOS"] = scenario
    env["EVAL_SEED"] = str(seed)
    env["RESULTS_TAG"] = run_tag
    cmd = [sys.executable, testscript_path]
    print(f"Running {run_tag}: seed={seed}, model={model}, scenario={scenario}")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Run single-constellation testscript over multiple seeds and keep per-run CSV outputs."
    )
    parser.add_argument(
        "--models",
        #default="PPO,DQN,BASELINE,ODT_FINETUNED",
        default = "ODT_FINETUNED",
        help="Comma-separated model labels used by testscript.",
    )
    parser.add_argument("--scenario", default="load_cycle_1", help="Single scenario to evaluate.")
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45,46,47,48,49,50",
        help="Comma-separated seeds (mapped to sim1, sim2, ...).",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_results_dir = os.path.dirname(this_dir)
    testscript_path = os.path.join(test_results_dir, "testscript.py")
    if not os.path.exists(testscript_path):
        raise FileNotFoundError(f"Could not find testscript at {testscript_path}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided. Pass --models with at least one model.")
    if not seeds:
        raise ValueError("No seeds provided. Pass --seeds with at least one seed.")

    # For each simulation seed, run all models with the SAME seed.
    generated = []
    for sim_idx, seed in enumerate(seeds, start=1):
        run_tag = f"sim{sim_idx}"
        for model in models:
            _run_once(testscript_path, model, args.scenario, seed, run_tag)
            out_csv = os.path.join(
                test_results_dir, f"{model}_observations_{args.scenario}_{run_tag}.csv"
            )
            generated.append((model, seed, run_tag, out_csv))

    print("\nGenerated run files:")
    for model, seed, run_tag, out_csv in generated:
        print(f"- {model} | {run_tag} | seed={seed} | {out_csv}")


if __name__ == "__main__":
    main()
