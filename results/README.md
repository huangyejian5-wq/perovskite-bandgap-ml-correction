# Results

This directory stores compact public-release result tables.

## Included Table

- `mp_unique_formula_pool.csv`
  - Description: unique-formula summary table grouped from `data/mp_screening_results.csv`, with model predictions added
  - Row count: 4511
  - Grouping key: `pretty_formula`
  - Representative row rule: the lowest-formation-energy entry is kept for each formula
  - Purpose: provides the unique Materials Project formula pool together with predicted bandgap values

## Notes

- This public release does not keep the downstream screened-candidate tables here.
- The 4511-row table is the unique-formula view of the original 6380-row Materials Project screening source.
- The `pred_Eg` and `pred_Eg_std` columns are generated from the released ExtraTrees screening workflow.
