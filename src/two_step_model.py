import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .data_pipeline import load_training_set, sanitize_columns, build_feature_matrix
from .gga_filter import is_metal_by_exp
from .physical_bounds import apply_lower_bound

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'


def two_step_oof_predictions(X, y_true, clf, reg, threshold=0.05, bounds_margin=None, gga_gap=None):
    y_cls = is_metal_by_exp(y_true, threshold=threshold).astype(int)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cls = y_cls[train_idx]
        y_train_reg = y_true[train_idx]

        clf.fit(X_train, y_train_cls)
        pred_cls = clf.predict(X_test)

        mask_train_nm = y_train_cls == 0
        if mask_train_nm.sum() > 0:
            reg.fit(X_train[mask_train_nm], y_train_reg[mask_train_nm])
            pred_reg = reg.predict(X_test)
        else:
            pred_reg = np.zeros(len(X_test))

        fold_pred = np.where(pred_cls == 1, 0.0, pred_reg)
        fold_pred = np.clip(fold_pred, 0.0, None)

        if bounds_margin is not None and gga_gap is not None:
            nm_mask = pred_cls == 0
            fold_pred = apply_lower_bound(fold_pred, gga_gap[test_idx], margin=bounds_margin, mask=nm_mask)

        y_pred[test_idx] = fold_pred

    return y_pred


def compare_models():
    df = load_training_set()
    df = sanitize_columns(df)

    drop_cols = ['Formula', 'E_g_Exp', 'Source', 'Priority', 'pretty_formula', 'Delta_E_g', 'is_metal_exp', 'target_delta']
    X, _ = build_feature_matrix(df, drop_cols)
    y = df['E_g_Exp'].values

    models = {
        'Extra Trees': (
            ExtraTreesClassifier(n_estimators=150, max_depth=10, random_state=42),
            ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        ),
    }

    try:
        import xgboost as xgb
        models['XGBoost'] = (
            xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.01, random_state=42, eval_metric='logloss'),
            xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42),
        )
    except Exception:
        pass

    try:
        import lightgbm as lgb
        models['LightGBM'] = (
            lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1),
            lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        )
    except Exception:
        pass

    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    models['Gradient Boosting'] = (
        GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    )

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    models['SVM / SVR'] = (
        make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)),
        make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0)),
    )

    models['Linear (Ridge/LogReg)'] = (
        make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    )

    rows = []
    for name, (clf, reg) in models.items():
        pred = two_step_oof_predictions(X, y, clf, reg)
        rows.append({'Algorithm': name, 'MAE': mean_absolute_error(y, pred), 'R2': r2_score(y, pred)})

    res = pd.DataFrame(rows).sort_values('R2', ascending=False)
    print(res.to_string(index=False))


def ablation():
    df = load_training_set()
    df = sanitize_columns(df)
    df['gap_vol_ratio'] = df['band_gap'] / (df['volume'] + 1e-5)

    drop_cols = ['Formula', 'E_g_Exp', 'Source', 'Priority', 'pretty_formula', 'Delta_E_g', 'is_metal_exp', 'target_delta']

    def run(use_two_step, use_nlp, use_features, use_bounds):
        data = df.copy() if use_nlp else df.sample(n=210, random_state=42)
        X, cols = build_feature_matrix(data, drop_cols)
        if not use_features and 'gap_vol_ratio' in cols:
            X = X.drop(columns=['gap_vol_ratio'])
        y = data['E_g_Exp'].values
        gga = data['band_gap'].values
        y_cls = is_metal_by_exp(y, threshold=0.05).astype(int)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        pred = np.zeros(len(X))

        if not use_two_step:
            reg = ExtraTreesRegressor(n_estimators=800, max_depth=20, random_state=42)
            for train_idx, test_idx in cv.split(X):
                reg.fit(X.iloc[train_idx], y[train_idx])
                fold = reg.predict(X.iloc[test_idx])
                pred[test_idx] = np.clip(fold, 0.0, None)
            return mean_absolute_error(y, pred), r2_score(y, pred)

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cls = y_cls[train_idx]
            y_train_reg = y[train_idx]

            clf = ExtraTreesClassifier(n_estimators=800, max_depth=20, random_state=42)
            if use_features:
                smote = SMOTE(random_state=42)
                try:
                    X_train_smote, y_train_cls_smote = smote.fit_resample(X_train, y_train_cls)
                    clf.fit(X_train_smote, y_train_cls_smote)
                except Exception:
                    clf.fit(X_train, y_train_cls)
            else:
                clf.fit(X_train, y_train_cls)

            pred_cls = clf.predict(X_test)
            mask_train_nm = y_train_cls == 0
            if mask_train_nm.sum() > 0:
                if use_features:
                    reg = ExtraTreesRegressor(
                        n_estimators=800,
                        max_depth=20,
                        random_state=42,
                    )
                else:
                    reg = ExtraTreesRegressor(n_estimators=200, random_state=42)
                reg.fit(X_train[mask_train_nm], y_train_reg[mask_train_nm])
                pred_reg = reg.predict(X_test)
            else:
                pred_reg = np.zeros(len(X_test))

            fold_pred = np.where(pred_cls == 1, 0.0, pred_reg)
            if use_bounds:
                nm_mask = pred_cls == 0
                fold_pred[nm_mask] = np.maximum(fold_pred[nm_mask], gga[test_idx][nm_mask] - 0.1)
            pred[test_idx] = np.clip(fold_pred, 0.0, None)

        return mean_absolute_error(y, pred), r2_score(y, pred)

    steps = [
        ('Single-step baseline (N=210)', dict(use_two_step=False, use_nlp=False, use_features=False, use_bounds=False)),
        ('+ Two-step workflow (N=210)', dict(use_two_step=True, use_nlp=False, use_features=False, use_bounds=False)),
        ('+ NLP augmentation (N=267)', dict(use_two_step=True, use_nlp=True, use_features=False, use_bounds=False)),
        ('+ Feature engineering + bounds (N=267)', dict(use_two_step=True, use_nlp=True, use_features=True, use_bounds=True)),
    ]

    for label, cfg in steps:
        mae, r2 = run(**cfg)
        print(f'{label}: MAE={mae:.3f} eV, R2={r2:.3f}')


def megnet_compare():
    model_dir = Path(os.getenv('MEGNET_MODEL_DIR', ''))
    if not model_dir:
        raise RuntimeError('MEGNET_MODEL_DIR is not set')

    cif_dir = DATA_DIR / 'cifs'
    if not cif_dir.exists():
        raise FileNotFoundError('data/cifs not found')

    import torch
    from pymatgen.core import Structure
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    from matgl.models import MEGNet

    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['DGL_HOME'] = str(ROOT_DIR / '.dgl')
    Path(os.environ['DGL_HOME']).mkdir(parents=True, exist_ok=True)

    with open(model_dir / 'model.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    model = MEGNet(**config['kwargs']['model']['init_args'])
    state_dict = torch.load(model_dir / 'model.pt', map_location='cpu')
    clean = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(clean, strict=False)

    cif_files = list(cif_dir.glob('*.cif'))
    if not cif_files:
        raise FileNotFoundError('No CIFs found under data/cifs')

    elem_list = get_element_list([Structure.from_file(p) for p in cif_files])
    converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

    df = load_training_set()
    df = sanitize_columns(df)
    df_nm = df[df['E_g_Exp'] > 0.05].reset_index(drop=True)

    embeddings = []
    valid = []
    for idx, row in df_nm.iterrows():
        formula = row['pretty_formula']
        cif_path = cif_dir / f'{formula}.cif'
        if not cif_path.exists():
            continue
        try:
            graph, lattice, state = converter.get_graph(Structure.from_file(cif_path))
            with torch.no_grad():
                node_feat = model.embedding.layer_node_embedding(graph.ndata['node_type'])
                embed = torch.mean(node_feat, dim=0).cpu().numpy()
            embeddings.append(embed)
            valid.append(idx)
        except Exception:
            continue

    if not embeddings:
        raise RuntimeError('Failed to extract any embeddings')

    X_megnet = np.vstack(embeddings)
    df_valid = df_nm.iloc[valid].reset_index(drop=True)

    drop_cols = ['Formula', 'E_g_Exp', 'Source', 'Priority', 'pretty_formula', 'Delta_E_g', 'is_metal_exp', 'target_delta']
    X_magpie, _ = build_feature_matrix(df_valid, drop_cols)
    y = df_valid['E_g_Exp'].values

    X_magpie = X_magpie.values
    X_combined = np.hstack([X_magpie, X_megnet])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_o, r2_o, mae_m, r2_m, mae_c, r2_c = [], [], [], [], [], []

    for train_idx, test_idx in cv.split(X_magpie):
        y_train, y_test = y[train_idx], y[test_idx]

        et1 = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
        et1.fit(X_magpie[train_idx], y_train)
        p1 = et1.predict(X_magpie[test_idx])
        mae_o.append(mean_absolute_error(y_test, p1))
        r2_o.append(r2_score(y_test, p1))

        et2 = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
        et2.fit(X_megnet[train_idx], y_train)
        p2 = et2.predict(X_megnet[test_idx])
        mae_m.append(mean_absolute_error(y_test, p2))
        r2_m.append(r2_score(y_test, p2))

        et3 = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
        et3.fit(X_combined[train_idx], y_train)
        p3 = et3.predict(X_combined[test_idx])
        mae_c.append(mean_absolute_error(y_test, p3))
        r2_c.append(r2_score(y_test, p3))

    print('=== Real MEGNet Embeddings Ablation Results ===')
    print(f'Proposed (Magpie only)         - MAE: {np.mean(mae_o):.3f} eV, R2: {np.mean(r2_o):.3f}')
    print(f'Scheme A1 (MEGNet Embed only)  - MAE: {np.mean(mae_m):.3f} eV, R2: {np.mean(r2_m):.3f}')
    print(f'Scheme A2 (Magpie + MEGNet)    - MAE: {np.mean(mae_c):.3f} eV, R2: {np.mean(r2_c):.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--megnet-compare', action='store_true')
    args = parser.parse_args()

    if args.compare:
        compare_models()
    if args.ablation:
        ablation()
    if args.megnet_compare:
        megnet_compare()


if __name__ == '__main__':
    main()
