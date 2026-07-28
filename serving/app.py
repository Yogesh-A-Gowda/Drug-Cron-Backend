from huggingface_hub import snapshot_download
import pandas as pd
import torch
import joblib
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

@app.on_event("startup")
async def startup():
    print("Pulling latest model from HF Hub...")
    model_dir = snapshot_download(repo_id="yogeshagowda/drug-distilbert", token=os.environ.get("HF_TOKEN"))
    app.state.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    app.state.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    app.state.model.eval()
    app.state.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
    # rest unchanged — load CSV, build drug_list, etc.

# --- CONFIG ---
MODEL_DIR = "./fine_tuned_distilbert_drug_reviews"
RAW_DATA = "drug_reviews_imputed_rf.csv"

app = FastAPI()

# Allow your laptop to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup():
    print("🚀 Booting BERT Engine...")
    app.state.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    app.state.model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    app.state.model.eval()
    app.state.label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    
    # Load Data
    df = pd.read_csv(RAW_DATA)
    df.columns = df.columns.str.lower()
    if 'drug_review' in df.columns: df.rename(columns={'drug_review': 'review_text'}, inplace=True)
    
    app.state.raw_df = df
    app.state.drug_list = sorted(df['drug_name'].astype(str).unique().tolist())
    print("✅ System Ready.")

def predict_batch(texts):
    """Speeds up inference by 5x using Batching instead of .apply()"""
    inputs = app.state.tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128 # Shorter length = Much faster CPU speed
    ).to(device)
    
    with torch.no_grad():
        outputs = app.state.model(**inputs)
    
    indices = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return app.state.label_encoder.inverse_transform(indices)

@app.get("/drugs")
def get_drugs():
    return app.state.drug_list

# @app.get("/analysis/{drug_name}")
# def analyze(drug_name: str):
#     df = app.state.raw_df
#     drug_df = df[df['drug_name'].str.lower() == drug_name.lower()].copy()
    
#     if drug_df.empty:
#         raise HTTPException(status_code=404, detail="Drug not found")

#     # Limit to 50 reviews to ensure the API stays under the timeout limit
#     drug_df = drug_df.head(50) 
    
#     # Batch Predict
#     texts = drug_df['review_text'].astype(str).tolist()
#     drug_df['predicted_category'] = predict_batch(texts)
    
#     # Weights for Trust Score
#     weights = {
#         "Positive_Experience": 1.0, "Mixed_Feedback": 0.0, "Ineffective": -0.7, 
#         "Dosage_Issues": -0.4, "Severe_Side_Effects": -0.9, "Dependency/Addiction": -1.0
#     }
#     drug_df['score_val'] = drug_df['predicted_category'].map(weights)
#     trust_score = (drug_df['score_val'].mean() + 1) / 2
    
#     return {
#         "drug_name": drug_name,
#         "trust_score": round(float(trust_score), 3),
#         "total_reviews": len(drug_df),
#         "review_summary": drug_df['predicted_category'].value_counts().to_dict(),
#         "all_classified_reviews": drug_df.to_dict('records')
#     }

@app.get("/analysis/{drug_name}")
def analyze(drug_name: str, skip: int = 0, limit: int = 50):
    df = app.state.raw_df
    # Filter for the specific drug
    drug_df_full = df[df['drug_name'].str.lower() == drug_name.lower()].copy()
    
    if drug_df_full.empty:
        raise HTTPException(status_code=404, detail="Drug not found")

    # This is where the 451 total comes from
    total_database_count = len(drug_df_full) 

    # Slice the dataframe based on pagination parameters
    drug_df_paged = drug_df_full.iloc[skip : skip + limit].copy()
    
    # Batch Predict only the current visible subset (e.g., 50 reviews)
    texts = drug_df_paged['review_text'].astype(str).tolist()
    drug_df_paged['predicted_category'] = predict_batch(texts)
    
    # Calculate Trust Score
    weights = {
        "Positive_Experience": 1.0, "Mixed_Feedback": 0.0, "Ineffective": -0.7, 
        "Dosage_Issues": -0.4, "Severe_Side_Effects": -0.9, "Dependency/Addiction": -1.0
    }
    drug_df_paged['score_val'] = drug_df_paged['predicted_category'].map(weights)
    trust_score = (drug_df_paged['score_val'].mean() + 1) / 2
    
    return {
        "drug_name": drug_name,
        "trust_score": round(float(trust_score), 3),
        "total_reviews": total_database_count, # The global total (451)
        "showing": len(drug_df_paged),
        "review_summary": drug_df_paged['predicted_category'].value_counts().to_dict(),
        "all_classified_reviews": drug_df_paged.to_dict('records')
    }
