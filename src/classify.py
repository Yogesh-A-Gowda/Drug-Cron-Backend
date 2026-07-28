"""Unchanged from your notebook — the keyword rule that generates training labels."""
import argparse
import pandas as pd

def classify_review_category(text):
    text = str(text).lower().replace("’", "'")
    positive_kw = ['helped', 'working', 'effective', 'relief', 'improved', 'better', 'great', 'life changing', 'happy']
    ineffective_kw = ['not work', 'no effect', 'useless', 'ineffective', 'waste of money', 'nothing happened', 'did nothing']
    dependency_kw = ['addicted', 'withdrawal', "can't stop", 'dependent', 'shakes', 'craving', 'addiction']
    dosage_kw = ['dosage', 'dose', 'too strong', 'too weak', '10mg', '20mg', '50mg', '100mg', 'mg', 'milligrams']
    side_effect_kw = ['nausea', 'vomit', 'dizzy', 'rash', 'insomnia', 'headache', 'tired', 'palpitations']

    if any(k in text for k in dependency_kw): return "Dependency/Addiction"
    if any(k in text for k in side_effect_kw):
        return "Mixed_Feedback" if any(k in text for k in positive_kw) else "Severe_Side_Effects"
    if any(k in text for k in dosage_kw): return "Dosage_Issues"
    if any(k in text for k in ineffective_kw): return "Ineffective"
    if any(k in text for k in positive_kw): return "Positive_Experience"
    return "Mixed_Feedback"

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.dropna(subset=['review_text'], inplace=True)
    df['review_category'] = df['review_text'].apply(classify_review_category)
    df.to_csv(output_csv, index=False)
    print(f"Labeled {len(df)} rows, saved to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.input, args.output)
