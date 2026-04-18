# Data

## Files

- `training_set_267.csv`: 267-sample curated training set with engineered features.
- `mp_screening_results.csv`: screening results table for Materials Project candidates.
- `manual_collection_source.xlsx`: primary manually curated source workbook.
- `manually_collected_data_english.csv`: public CSV export translated to English for easier scripting and versioning.
- `experimental_bandgap_metadata_curated.csv`: normalized metadata table rebuilt from the manual source workbook.
- `experimental_bandgap_metadata_cleaned.csv`: stricter review table rebuilt from the manual source workbook, with explicit quality flags and manual-review markers.
- `final_training_samples.csv`: earlier-stage sample subset (used by some ablation steps).
- `ultimate_experimental_ground_truth.csv`: experimental bandgap ground-truth table.
- `new_extracted_bandgaps.csv`: older record table kept for reference only; it is no longer the primary manual source.
- `cifs/`: optional CIF files named as `<pretty_formula>.cif` when prepared separately.

## Columns (experimental_bandgap_metadata_curated.csv)

- `formula`: perovskite formula
- `compound_class`: single/double perovskite type label from the manual workbook
- `formula_standardized`: normalized formula confirmed during manual entry
- `reported_bandgap_values_raw`: raw reported value string kept from the manual collection table
- `crystal_structure`: reported crystal structure or phase label
- `experimental_bandgap_eV`: manually curated experimental bandgap value in eV
- `bandgap_character`: direct / indirect / unknown
- `measurement_method`: reported measurement method
- `measurement_temperature`: reported measurement temperature
- `sample_form`: sample morphology or specimen form
- `is_bulk`: bulk / non-bulk / unclear flag
- `doi_or_url`: left blank in the public release
- `first_author`: first-author field from the manual workbook
- `year`: publication year
- `title`: article title
- `confidence_grade`: manual confidence grade
- `gga_gap_reference`: optional GGA comparison note
- `curation_note`: manual note entered during curation
- `entry_info`: curator/date entry
- `metadata_source`: source tag used during table assembly

## Columns (experimental_bandgap_metadata_cleaned.csv)

- `formula`: formula recorded in the manual workbook
- `reported_bandgap_values_raw`: raw value string from the source record
- `experimental_bandgap_eV`: curated experimental bandgap value
- `record_type`: source category from manual curation, e.g. `experimental`, `theoretical`, `review`
- `is_experimental`: simplified yes/no/unclear flag
- `formula_content_match`: manual confirmation of whether the paper is centered on the target formula
- `needs_manual_review`: yes/no flag for records that still require manual inspection
- `quality_flags`: machine-readable review flags derived from the manual workbook

## Columns (training_set_267.csv)

Key columns used by the scripts:

- `pretty_formula`: identifier used to match CIF files
- `E_g_Exp`: experimental bandgap (target)
- `band_gap`: GGA bandgap (baseline)
- feature columns: numeric engineered descriptors

## CIFs

CIFs are not included by default in the public release.
