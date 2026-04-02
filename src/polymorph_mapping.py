import pandas as pd


def select_formula_representatives(df, formula_col="Formula"):
    if formula_col not in df.columns:
        raise KeyError(f"{formula_col} not found in dataframe")
    return df.drop_duplicates(subset=[formula_col]).reset_index(drop=True)


def summarize_formula_counts(df, formula_col="Formula"):
    if formula_col not in df.columns:
        raise KeyError(f"{formula_col} not found in dataframe")
    counts = df[formula_col].value_counts().rename_axis(formula_col).reset_index(name="count")
    return counts
