# perovskite-bandgap-ml-correction

Machine-learning correction of GGA bandgap errors in inorganic perovskites, including:

- Two-step classification → regression workflow for metal vs non-metal handling
- Ablation study reproduction
- Optional GNN embedding comparison (MEGNet)

## Layout

- `data/`: datasets and screening results
- `src/`: core pipeline modules
- `notebooks/`: example notebooks (reproducibility walkthrough)
- `results/models/`: placeholder for trained model artifacts

## Quickstart

Create an environment and install core dependencies:

```bash
pip install -r requirements.txt
```

Run the main baseline comparison (257-sample dataset):

```bash
python -m src.two_step_model --compare
```

Run the manuscript ablation workflow:

```bash
python -m src.two_step_model --ablation
```

## GNN comparison (optional)

The MEGNet comparison requires prepared CIF files under `data/cifs/` and a compatible deep-learning stack.

```bash
export MEGNET_MODEL_DIR=/path/to/MEGNet-MP-2019.4.1-BandGap-mfi
python -m src.two_step_model --megnet-compare
```

## Notes

- This repository does not include crawler utilities or plotting scripts in the public release.
- Some heavy dependencies (e.g., DGL) may require platform-specific installation.
