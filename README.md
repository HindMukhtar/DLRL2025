# DLRL2025 - Single Constellation

LEO satellite handover simulation and RL/ODT training pipeline for aircraft connectivity.

## Project Summary

This project includes:
- A simulation environment for aircraft-to-LEO beam connectivity.
- RL baselines (`PPO`, `DQN`, `BASELINE`).
- Offline dataset generation for ODT.
- Offline ODT training and online fine-tuning.
- Evaluation scripts and result-analysis notebooks.

Core optimization goals:
- Maximize allocation-to-demand ratio / throughput.
- Minimize service drop.
- Minimize latency and latency violations.

## Folder Structure

- `Environment/`
  - `LEOEnvironment.py`: base simulation environment.
  - `LEOEnvironmentRL.py`: RL-oriented environment logic.

- `Training/`
  - `HandoverEnvironment.py`: PPO training environment wrapper/script.
  - `HandoverEnvironment_DQN.py`: DQN training environment wrapper/script.
  - `ODT.py`: ODT model + ODT env wrapper + inference helpers.
  - `train_odt_offline.py`: offline ODT training.
  - `odt_online_finetune.py`: online ODT fine-tuning.
  - `sweep_odt_target_return.py`: RTG target sweep utility.

- `dataset/`
  - `generate_odt_dataset.py`: trajectory generation/augmentation.
  - `*.pkl`: offline dataset artifacts.

- `models/`
  - Trained PPO/DQN/ODT checkpoints (`.zip`, `.pth`).

- `test results/`
  - `testscript.py`: evaluation runner for agents/scenarios.
  - `testscript.ipynb`: plotting/inspection notebook.
  - `*_observations_*.csv`, `*_summary_*.pkl`: evaluation outputs.

- `routes/`
  - Route source/interpolated files and interpolation utility.

- `Inputs/`
  - `input.csv`: global run configuration (e.g., constellation/test length).
  - `PopMap_500.png`: population map used for load modeling.

- `Analysis/`
  - Helper scripts for beam availability and GIF generation.

- `Archive scripts/`
  - Historical/legacy scripts kept for reference.

## Typical Workflow

1. Generate or augment dataset:
   - `python3 "dataset/generate_odt_dataset.py"`

2. Train offline ODT:
   - `python3 "Training/train_odt_offline.py"`

3. Fine-tune ODT online:
   - `python3 "Training/odt_online_finetune.py"`

4. Evaluate models:
   - `python3 "test results/testscript.py"`

5. Analyze outputs:
   - `paper_results_summary.ipynb`
   - `test results/testscript.ipynb`

## Dependencies

Install dependencies from:
- `requirements.txt`

Example:
- `python3 -m pip install -r "requirements.txt"`
