import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'


def load_training_set(path=None):
    path = Path(path) if path else (DATA_DIR / 'training_set_257.csv')
    return pd.read_csv(path)


def sanitize_columns(df):
    df = df.copy()
    df.rename(columns=lambda x: re.sub(r'[\[\]<]', '_', x), inplace=True)
    return df


def build_feature_matrix(df, drop_cols):
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in [np.float64, np.int64]]
    X = df[feature_cols].fillna(df[feature_cols].mean()).fillna(0)
    return X, feature_cols


def get_unlabeled_formulas():
    predicted = DATA_DIR / 'mp_screening_results.csv'
    training = DATA_DIR / 'final_201_training_samples.csv'
    if not predicted.exists():
        return []
    df_pred = pd.read_csv(predicted)
    if 'pretty_formula' not in df_pred.columns:
        return []
    candidates = set(df_pred['pretty_formula'].dropna().unique())
    if training.exists():
        df_train = pd.read_csv(training)
        if 'Formula' in df_train.columns:
            candidates -= set(df_train['Formula'].dropna().unique())
    return list(candidates)


def extract_bandgap_from_text(text):
    if not text:
        return []
    matches = re.findall(r'(\d+\.\d+|\d+)\s*[eE][vV]', text)
    return list(dict.fromkeys(matches))


def rebuild_abstract(abstract_inverted):
    if not abstract_inverted:
        return ''
    words = []
    for word, positions in abstract_inverted.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return ' '.join(word for _, word in words)


def extract_first_float(value):
    try:
        if pd.isna(value):
            return np.nan
        first = str(value).split(',')[0].strip()
        result = float(first)
        return result if 0 <= result <= 20 else np.nan
    except Exception:
        return np.nan


def merge_bandgaps():
    df_new_path = DATA_DIR / 'new_extracted_bandgaps.csv'
    df_gt_path = DATA_DIR / 'ultimate_experimental_ground_truth.csv'
    if not df_new_path.exists() or not df_gt_path.exists():
        raise FileNotFoundError('Missing new_extracted_bandgaps.csv or ultimate_experimental_ground_truth.csv')

    df_new = pd.read_csv(df_new_path)
    df_new['E_g_Exp'] = df_new['Extracted_Eg_eV'].apply(extract_first_float)
    df_new = df_new.dropna(subset=['E_g_Exp'])

    df_new_formatted = pd.DataFrame(
        {
            'Formula': df_new['Formula'],
            'E_g_Exp': df_new['E_g_Exp'],
            'Source': 'external_record_merge',
            'Priority': 2,
        }
    )

    df_orig = pd.read_csv(df_gt_path)
    df_merged = pd.concat([df_orig, df_new_formatted], ignore_index=True)
    df_merged = df_merged.sort_values('Priority').drop_duplicates(subset=['Formula'], keep='first')
    df_merged.to_csv(df_gt_path, index=False)
    print('Saved merged dataset to data/ultimate_experimental_ground_truth.csv')
