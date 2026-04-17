from pathlib import Path
import re

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.data_pipeline import build_feature_matrix, sanitize_columns
from src.gga_filter import is_metal_by_exp


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "notebooks" / "origin_data"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_number(text):
    s = str(text)
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def load_df_267():
    df = pd.read_csv(DATA_DIR / "training_set_257.csv")
    df = sanitize_columns(df)
    df["gap_vol_ratio"] = df["band_gap"] / (df["volume"] + 1e-5)
    return df


def load_df_98():
    df = pd.read_csv(DATA_DIR / "final_201_training_samples.csv")
    df = sanitize_columns(df)
    df = df.sample(n=98, random_state=42).reset_index(drop=True)
    return df


def load_ngga_baseline_115():
    meta = pd.read_csv(DATA_DIR / "experimental_bandgap_metadata_cleaned.csv")
    mask = (
        meta["confidence_grade"].astype(str).isin(["A", "B"])
        & meta["needs_manual_review"].astype(str).eq("no")
        & meta["record_type"].astype(str).eq("experimental")
        & meta["compound_class"].astype(str).str.contains("single|double", case=False, na=False)
    )
    sub = meta.loc[mask].copy()
    # Prefer the curated band_gap in the final table; fallback to metadata GGA reference.
    train = pd.read_csv(DATA_DIR / "training_set_257.csv")[["Formula", "band_gap"]]
    merged = sub.merge(train, left_on="formula_standardized", right_on="Formula", how="left")
    merged["gga_ref_num"] = merged["gga_gap_reference"].apply(parse_number)
    merged["gga_final"] = merged["band_gap"].combine_first(merged["gga_ref_num"])
    merged["E_g_Exp"] = pd.to_numeric(merged["experimental_bandgap_eV"], errors="coerce")
    merged = merged.dropna(subset=["E_g_Exp", "gga_final"]).copy()
    return merged


def build_X(df, extra_feature=False):
    drop_cols = [
        "Formula",
        "E_g_Exp",
        "Source",
        "Priority",
        "pretty_formula",
        "Delta_E_g",
        "is_metal_exp",
        "target_delta",
    ]
    X, cols = build_feature_matrix(df, drop_cols)
    if not extra_feature and "gap_vol_ratio" in X.columns:
        X = X.drop(columns=["gap_vol_ratio"])
    return X


def eval_single_step(df, extra_feature=False):
    X = build_X(df, extra_feature=extra_feature)
    y = df["E_g_Exp"].values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(df), dtype=float)
    for tr, te in cv.split(X):
        reg = ExtraTreesRegressor(n_estimators=800, max_depth=20, random_state=42, n_jobs=-1)
        reg.fit(X.iloc[tr], y[tr])
        fold = reg.predict(X.iloc[te])
        pred[te] = np.clip(fold, 0.0, None)
    return pred


def eval_two_step(df, extra_feature=False, use_bounds=False):
    X = build_X(df, extra_feature=extra_feature)
    y = df["E_g_Exp"].values
    gga = df["band_gap"].values
    y_cls = is_metal_by_exp(y, threshold=0.05).astype(int)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(df), dtype=float)

    for tr, te in cv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr_cls = y_cls[tr]
        y_tr = y[tr]

        clf = ExtraTreesClassifier(n_estimators=800, max_depth=20, random_state=42, n_jobs=-1)
        if extra_feature:
            try:
                smote = SMOTE(random_state=42)
                X_sm, y_sm = smote.fit_resample(X_tr, y_tr_cls)
                clf.fit(X_sm, y_sm)
            except Exception:
                clf.fit(X_tr, y_tr_cls)
        else:
            clf.fit(X_tr, y_tr_cls)

        pred_cls = clf.predict(X_te)
        mask_nm = y_tr_cls == 0
        if mask_nm.sum() > 0:
            reg = ExtraTreesRegressor(
                n_estimators=800 if extra_feature else 200,
                max_depth=20 if extra_feature else None,
                random_state=42,
                n_jobs=-1,
            )
            reg.fit(X_tr[mask_nm], y_tr[mask_nm])
            pred_reg = reg.predict(X_te)
        else:
            pred_reg = np.zeros(len(X_te))

        fold_pred = np.where(pred_cls == 1, 0.0, pred_reg)
        if use_bounds:
            nm_mask = pred_cls == 0
            fold_pred[nm_mask] = np.maximum(fold_pred[nm_mask], gga[te][nm_mask] - 0.1)
        pred[te] = np.clip(fold_pred, 0.0, None)

    return pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # NGGA baseline on the closest recoverable 115-scheme subset (113 usable entries after GGA-reference merge).
    df_115 = load_ngga_baseline_115()
    y_115 = df_115["E_g_Exp"].values
    pred_115 = df_115["gga_final"].values
    rows.append(
        {
            "Scheme": "NGGA Baseline",
            "Description": "DFT direct calculation",
            "Nominal_N": 115,
            "Effective_N": len(df_115),
            "MAE": mean_absolute_error(y_115, pred_115),
            "RMSE": rmse(y_115, pred_115),
            "R2": r2_score(y_115, pred_115),
        }
    )

    # Scheme A: pure ground-state mapping, single-step regression, 98 samples.
    df_98 = load_df_98()
    y_98 = df_98["E_g_Exp"].values
    pred_98 = eval_single_step(df_98, extra_feature=False)
    rows.append(
        {
            "Scheme": "Scheme A",
            "Description": "Pure ground-state mapping, single-step regression",
            "Nominal_N": 98,
            "Effective_N": len(df_98),
            "MAE": mean_absolute_error(y_98, pred_98),
            "RMSE": rmse(y_98, pred_98),
            "R2": r2_score(y_98, pred_98),
        }
    )

    # Scheme B/C/D on 267-sample final table.
    df_267 = load_df_267()
    y_267 = df_267["E_g_Exp"].values

    pred_B = eval_single_step(df_267, extra_feature=False)
    rows.append(
        {
            "Scheme": "Scheme B",
            "Description": "Thermodynamic relaxation + physical filtering, single-step regression",
            "Nominal_N": 267,
            "Effective_N": len(df_267),
            "MAE": mean_absolute_error(y_267, pred_B),
            "RMSE": rmse(y_267, pred_B),
            "R2": r2_score(y_267, pred_B),
        }
    )

    pred_D = eval_two_step(df_267, extra_feature=True, use_bounds=True)
    rows.append(
        {
            "Scheme": "Scheme C",
            "Description": "Complete framework (+ physical hard-bound constraints)",
            "Nominal_N": 267,
            "Effective_N": len(df_267),
            "MAE": mean_absolute_error(y_267, pred_D),
            "RMSE": rmse(y_267, pred_D),
            "R2": r2_score(y_267, pred_D),
        }
    )

    res = pd.DataFrame(rows)
    out = OUT_DIR / "Ablation_New_Schemes_Results.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
