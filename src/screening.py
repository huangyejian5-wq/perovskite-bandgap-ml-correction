import argparse
import os
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'


def fetch_cifs():
    api_key = os.getenv('MP_API_KEY')
    if not api_key:
        raise RuntimeError('MP_API_KEY is not set')

    df = pd.read_csv(DATA_DIR / 'training_set_257.csv')
    formulas = df['pretty_formula'].dropna().unique()
    save_dir = DATA_DIR / 'cifs'
    save_dir.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key) as mpr:
        for i, formula in enumerate(formulas, start=1):
            cif_path = save_dir / f'{formula}.cif'
            if cif_path.exists():
                continue
            docs = mpr.summary.search(formula=formula, fields=['material_id', 'structure', 'energy_above_hull'])
            if not docs:
                continue
            best_doc = sorted(docs, key=lambda x: x.energy_above_hull)[0]
            best_doc.structure.to(fmt='cif', filename=str(cif_path))
            print(f'[{i}/{len(formulas)}] {formula} {best_doc.material_id}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-cifs', action='store_true')
    args = parser.parse_args()
    if args.fetch_cifs:
        fetch_cifs()


if __name__ == '__main__':
    main()
