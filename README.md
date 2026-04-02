# perovskite-bandgap-ml-correction

Machine-learning correction of GGA bandgap errors in inorganic perovskites, including:

- Two-step classification → regression workflow for metal vs non-metal handling
- Ablation study reproduction
- Data acquisition utilities (literature bandgap crawling, MP CIF download)
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

## Data acquisition

Crawl candidate experimental bandgaps from OpenAlex:

```bash
python -m src.data_pipeline --crawl-openalex --num-formulas 10
python -m src.data_pipeline --merge-bandgaps
```

Download CIFs from Materials Project (optional):

```bash
export MP_API_KEY=YOUR_KEY
python -m src.screening --fetch-cifs
```

## GNN comparison (optional)

The MEGNet comparison requires CIFs under `data/cifs/` and a compatible deep-learning stack.

```bash
export MEGNET_MODEL_DIR=/path/to/MEGNet-MP-2019.4.1-BandGap-mfi
python -m src.two_step_model --megnet-compare
```

## Notes

- This repository does not ship manuscript figures.
- Some heavy dependencies (e.g., DGL) may require platform-specific installation.
