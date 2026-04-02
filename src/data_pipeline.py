import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

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
    return ' '.join(word for pos, word in words)


def crawl_openalex(num_formulas=10, max_results_per_formula=3, delay=0.5):
    formulas = get_unlabeled_formulas()[: int(num_formulas)]
    if not formulas:
        print('No formulas to process. Exiting.')
        return

    base_url = 'https://api.openalex.org/works'
    results = []

    print(f'Starting crawler for {len(formulas)} formulas...')
    for i, formula in enumerate(formulas, start=1):
        print(f'[{i}/{len(formulas)}] Searching for {formula}...')
        query = quote(f'"{formula}" AND ("bandgap" OR "band gap" OR "Eg" OR "experimental")')
        url = f'{base_url}?search={query}&per-page={max_results_per_formula}&sort=relevance_score:desc'

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            works = response.json().get('results', [])
        except Exception as exc:
            print(f'  -> Request failed for {formula}: {exc}')
            time.sleep(delay)
            continue

        for paper in works:
            title = paper.get('title', '')
            abstract = rebuild_abstract(paper.get('abstract_inverted_index', {}))
            values = extract_bandgap_from_text(f'{title} {abstract}')
            if not values:
                continue
            results.append(
                {
                    'Formula': formula,
                    'Extracted_Eg_eV': ', '.join(values),
                    'Title': title,
                    'Year': paper.get('publication_year', ''),
                    'DOI': paper.get('doi', ''),
                    'Abstract_Snippet': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                }
            )

        time.sleep(delay)

    if results:
        out = DATA_DIR / 'new_extracted_bandgaps.csv'
        pd.DataFrame(results).to_csv(out, index=False)
        print(f'Data saved to {out.name}')
    else:
        print('No bandgap data found for the selected formulas.')


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
            'Source': 'OpenAlex_Crawler',
            'Priority': 2,
        }
    )

    df_orig = pd.read_csv(df_gt_path)
    df_merged = pd.concat([df_orig, df_new_formatted], ignore_index=True)
    df_merged = df_merged.sort_values('Priority').drop_duplicates(subset=['Formula'], keep='first')
    df_merged.to_csv(df_gt_path, index=False)
    print('Saved merged dataset to data/ultimate_experimental_ground_truth.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl-openalex', action='store_true')
    parser.add_argument('--num-formulas', type=int, default=10)
    parser.add_argument('--merge-bandgaps', action='store_true')
    args = parser.parse_args()

    if args.crawl_openalex:
        crawl_openalex(num_formulas=args.num_formulas)
    if args.merge_bandgaps:
        merge_bandgaps()


if __name__ == '__main__':
    main()
