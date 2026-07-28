"""Adapted from Data_Imputation.ipynb — RF IterativeImputer version, parameterized."""
import argparse
import pandas as pd, numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_out = df.copy()

    df_out['review_len'] = df_out['review_text'].fillna("").str.len()
    df_out['drug_freq'] = df_out['drug_name'].map(df_out['drug_name'].value_counts()).fillna(0)
    df_out['source_freq'] = df_out['source'].map(df_out['source'].value_counts()).fillna(0)

    numeric_cols = ['age', 'weight', 'review_len', 'drug_freq', 'source_freq']
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                                max_iter=10, random_state=42)
    numeric_imputed = imputer.fit_transform(df_out[numeric_cols])
    numeric_imputed_df = pd.DataFrame(numeric_imputed, columns=numeric_cols)

    df_out['age'] = numeric_imputed_df['age'].round().astype(int).clip(lower=10, upper=100)
    df_out['weight'] = numeric_imputed_df['weight'].round(1).clip(lower=30, upper=200)

# Normalize ALL casing variants to a consistent Male/Female before counting
    df_out['gender'] = df_out['gender'].astype(str).str.strip().str.lower().replace({'m': 'male', 'f': 'female', 'nan': None, 'none': None })
    df_out['gender'] = df_out['gender'].map({'male': 'Male', 'female': 'Female'})

    dist = df_out['gender'].value_counts(normalize=True)
    p_male = dist.get("Male", 0.5)
    p_female = dist.get("Female", 0.5)

    # Defensive: guarantee these sum to exactly 1 regardless of stray categories
    total = p_male + p_female
    if total == 0:
        p_male, p_female = 0.5, 0.5
    else:
        p_male, p_female = p_male / total, p_female / total

    mask = df_out['gender'].isna()
    if mask.sum() > 0:
        df_out.loc[mask, 'gender'] = np.random.choice(["Male", "Female"], size=mask.sum(), p=[p_male, p_female])

    df_out.to_csv(output_csv, index=False)
    print(f"Imputed dataset saved to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.input, args.output)
