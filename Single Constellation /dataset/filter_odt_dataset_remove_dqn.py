import argparse
import os
import pickle


def default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".pkl"
    return f"{root}_new{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Remove DQN trajectories from an ODT dataset pickle."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(__file__),
            "odt_offline_dataset_aug_multisim.pkl",
        ),
        help="Path to input dataset pickle.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output dataset pickle. Defaults to input filename with _new suffix.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with open(input_path, "rb") as f:
        trajectories = pickle.load(f)

    if not isinstance(trajectories, list):
        raise RuntimeError("Expected dataset to be a list of trajectories.")

    filtered = [
        t for t in trajectories if str(t.get("source", "")).strip().lower() != "dqn"
    ]

    removed = len(trajectories) - len(filtered)
    with open(output_path, "wb") as f:
        pickle.dump(filtered, f)

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Total:   {len(trajectories)}")
    print(f"Removed: {removed}")
    print(f"Kept:    {len(filtered)}")


if __name__ == "__main__":
    main()
