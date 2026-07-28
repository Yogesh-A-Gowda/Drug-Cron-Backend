import os
import pandas as pd
import torch
import joblib
from huggingface_hub import snapshot_download
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIG ---
HF_MODEL_REPO = "yogeshagowda/drug-distilbert"
RAW_DATA = "drug_reviews_imputed_rf.csv"  # TODO: source this from HF dataset repo/S3 too — see note below

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup():
    print("Pulling latest model from HF Hub...")
    model_dir = snapshot_download(repo_id=HF_MODEL_REPO, token=os.environ.get("HF_TOKEN"))
    app.state.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    app.state.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    app.state.model.eval()
    app.state.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")

    print("Loading review data...")
    df = pd.read_csv(RAW_DATA)
    df.columns = df.columns.str.lower()
    if 'drug_review' in df.columns:
        df.rename(columns={'drug_review': 'review_text'}, inplace=True)

    app.state.raw_df = df
    app.state.drug_list = sorted(df['drug_name'].astype(str).unique().tolist())
    print("System ready.")

def predict_batch(texts):
    """Batched inference — much faster than .apply() on CPU."""
    inputs = app.state.tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = app.state.model(**inputs)
    indices = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return app.state.label_encoder.inverse_transform(indices)

@app.get("/drugs")
def get_drugs():
    return app.state.drug_list

@app.get("/analysis/{drug_name}")
def analyze(drug_name: str, skip: int = 0, limit: int = 50):
    df = app.state.raw_df
    drug_df_full = df[df['drug_name'].str.lower() == drug_name.lower()].copy()

    if drug_df_full.empty:
        raise HTTPException(status_code=404, detail="Drug not found")

    total_database_count = len(drug_df_full)
    drug_df_paged = drug_df_full.iloc[skip: skip + limit].copy()

    texts = drug_df_paged['review_text'].astype(str).tolist()
    drug_df_paged['predicted_category'] = predict_batch(texts)

    weights = {
        "Positive_Experience": 1.0, "Mixed_Feedback": 0.0, "Ineffective": -0.7,
        "Dosage_Issues": -0.4, "Severe_Side_Effects": -0.9, "Dependency/Addiction": -1.0
    }
    drug_df_paged['score_val'] = drug_df_paged['predicted_category'].map(weights)
    trust_score = (drug_df_paged['score_val'].mean() + 1) / 2

    return {
        "drug_name": drug_name,
        "trust_score": round(float(trust_score), 3),
        "total_reviews": total_database_count,
        "showing": len(drug_df_paged),
        "review_summary": drug_df_paged['predicted_category'].value_counts().to_dict(),
        "all_classified_reviews": drug_df_paged.to_dict('records')
    }
