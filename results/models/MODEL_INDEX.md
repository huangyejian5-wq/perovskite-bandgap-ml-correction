# Model Index

This directory documents the main model artifacts used in this repository.

No binary model files are shipped in the public release. Trained artifacts can be regenerated from the provided data and scripts.

## Core Models

- `two_step_classifier`
  - Task: metal vs non-metal preclassification
  - Implementation: `src/two_step_model.py`
  - Main algorithm: ExtraTreesClassifier
  - Input data: `data/training_set_257.csv`

- `two_step_regressor`
  - Task: bandgap correction on the non-metal branch
  - Implementation: `src/two_step_model.py`
  - Main algorithm: ExtraTreesRegressor
  - Input data: `data/training_set_257.csv`

- `screening_classifier`
  - Task: candidate prefiltering for Materials Project screening
  - Implementation: `screen_candidates.py`
  - Main algorithm: ExtraTreesClassifier
  - Input data: `data/training_set_257.csv`, `data/mp_screening_results.csv`

- `screening_regressor`
  - Task: candidate bandgap prediction for screened Materials Project compounds
  - Implementation: `screen_candidates.py`
  - Main algorithm: ExtraTreesRegressor
  - Input data: `data/training_set_257.csv`, `data/mp_screening_results.csv`

## Reproduction

- Main comparison: `python -m src.two_step_model --compare`
- Ablation study: `python run_new_ablation_schemes.py`
- Candidate screening: `python screen_candidates.py`

## Public Release Policy

- Included: scripts, datasets, and result tables
- Not included: serialized binary models, cache files, environment-specific temporary files
