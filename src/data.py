"""Load the German Credit data (1,000 borrowers) and save a copy to data/german_credit.csv.

Source: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository.
This 10-column version ships with the dalex Python package (dalex.datasets.load_german).
Target: risk = 1 good borrower, 0 bad borrower (defaulted).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "german_credit.csv"

TARGET = "risk"
# 'sex' is excluded on purpose: a bank should not price credit on gender, and in many
# countries it is not allowed to. It is kept in the data only to check fairness afterwards.
EXCLUDED = ["sex"]


def load() -> pd.DataFrame:
    if CSV.exists():
        return pd.read_csv(CSV)
    import dalex as dx
    df = dx.datasets.load_german()
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def split_xy(df: pd.DataFrame):
    y = (df[TARGET] == 0).astype(int)  # 1 = default, the event we want to predict
    X = df.drop(columns=[TARGET] + EXCLUDED)
    return X, y
