# Data

## Files

- `training_set_257.csv`: 257-sample curated training set with engineered features.
- `mp_screening_results.csv`: screening results table for Materials Project candidates.
- `final_201_training_samples.csv`: earlier-stage 201-sample subset (used by some ablation steps).
- `ultimate_experimental_ground_truth.csv`: experimental bandgap ground-truth table.
- `new_extracted_bandgaps.csv`: raw OpenAlex crawler outputs.
- `cifs/`: optional CIF files named as `<pretty_formula>.cif`.

## Columns (training_set_257.csv)

Key columns used by the scripts:

- `pretty_formula`: identifier used to match CIF files
- `E_g_Exp`: experimental bandgap (target)
- `band_gap`: GGA bandgap (baseline)
- feature columns: numeric engineered descriptors

## CIFs

CIFs are not included by default. Use `python -m src.screening --fetch-cifs` after setting `MP_API_KEY`.
