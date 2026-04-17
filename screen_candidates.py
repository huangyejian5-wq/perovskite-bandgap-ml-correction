from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from src.data_pipeline import sanitize_columns, build_feature_matrix

ROOT = Path(__file__).resolve().parent

# 1. Load Training Data
df_train = pd.read_csv(ROOT / 'data' / 'training_set_257.csv')
df_train = sanitize_columns(df_train)

drop_cols = ['Formula', 'E_g_Exp', 'Source', 'Priority', 'pretty_formula', 'Delta_E_g', 'is_metal_exp', 'target_delta']
X_train_df, feature_cols = build_feature_matrix(df_train, drop_cols)

X_train_full = X_train_df.values
y_reg_full = df_train['E_g_Exp'].values
y_cls_full = (y_reg_full <= 0.05).astype(int)

# 2. Train Final Models
clf_final = ExtraTreesClassifier(
    n_estimators=300,
    class_weight={0: 1, 1: 5},
    random_state=42, n_jobs=-1
)
clf_final.fit(X_train_full, y_cls_full)

reg_final = ExtraTreesRegressor(
    n_estimators=500,
    max_features=0.3,
    min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
# Train only on non-metals
non_metal_mask_train = (y_cls_full == 0)
reg_final.fit(X_train_full[non_metal_mask_train], y_reg_full[non_metal_mask_train])

# 3. Load Candidates (MP Database)
df_mp = pd.read_csv(ROOT / 'data' / 'mp_screening_results.csv')
df_mp = sanitize_columns(df_mp)

# Filter columns to match training set
missing_cols = [c for c in feature_cols if c not in df_mp.columns]
for c in missing_cols:
    df_mp[c] = 0.0

X_cand_df = df_mp[feature_cols].fillna(df_mp[feature_cols].mean()).fillna(0)
X_candidates = X_cand_df.values
formulas = df_mp['pretty_formula'].values
y_gga_candidates = df_mp['band_gap'].values

# 4. Predict
pred_cls_candidates = clf_final.predict(X_candidates)
non_metal_mask = (pred_cls_candidates == 0)

pred_eg = np.zeros(len(X_candidates))
if non_metal_mask.sum() > 0:
    pred_eg[non_metal_mask] = reg_final.predict(X_candidates[non_metal_mask])

# Physical bounds
pred_eg = np.clip(pred_eg, y_gga_candidates - 0.1, None)

# Uncertainty (tree variance)
tree_predictions = np.array([
    tree.predict(X_candidates[non_metal_mask])
    for tree in reg_final.estimators_
])
pred_std = tree_predictions.std(axis=0)

# 5. Screen Target Candidates (Semiconductors, 1.0 < Eg < 3.5)
target_mask = (non_metal_mask & (pred_eg > 1.0) & (pred_eg < 3.5))

candidates_df = pd.DataFrame({
    'formula': formulas[target_mask],
    'GGA_Eg': y_gga_candidates[target_mask],
    'pred_Eg': pred_eg[target_mask],
    'pred_Eg_std': pred_std[target_mask[non_metal_mask]],
    'is_false_metal_rescued': (y_gga_candidates[target_mask] < 0.1).astype(int)
})

# Filter out training set formulas to ensure true novelty
train_formulas = set(df_train['pretty_formula'].dropna().unique())
candidates_df = candidates_df[~candidates_df['formula'].isin(train_formulas)]

# Remove duplicates if any
candidates_df = candidates_df.drop_duplicates(subset=['formula'])

candidates_df = candidates_df.sort_values('pred_Eg_std')

print(f"Screened candidates: {len(candidates_df)}")
print("\nTop 10 high-confidence candidates (lowest predictive uncertainty):")
print(candidates_df.head(10)[['formula', 'GGA_Eg', 'pred_Eg', 'pred_Eg_std', 'is_false_metal_rescued']].to_string(index=False))

candidates_df.to_csv('results/screened_candidates_final.csv', index=False)
print("\nSaved all screened candidates to results/screened_candidates_final.csv")
