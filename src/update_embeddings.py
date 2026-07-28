"""New — generates embeddings for RAG and upserts into Neon pgvector."""
import argparse, os
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

def run(input_csv):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df['review_text'].astype(str).tolist(), show_progress_bar=True)

    conn = psycopg2.connect(os.environ["NEON_DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS review_embeddings (
            id SERIAL PRIMARY KEY,
            drug_name TEXT,
            review_text TEXT,
            predicted_category TEXT,
            embedding VECTOR(384)
        )
    """)
    for i, row in df.iterrows():
        cur.execute(
            "INSERT INTO review_embeddings (drug_name, review_text, predicted_category, embedding) VALUES (%s, %s, %s, %s)",
            (row['drug_name'], row['review_text'], row.get('predicted_category'), embeddings[i].tolist())
        )
    conn.commit()
    cur.close(); conn.close()
    print(f"Upserted {len(df)} embeddings into Neon")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    run(args.input)
