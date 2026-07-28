"""
Runs on Kaggle's infrastructure (GPU-enabled). Pulls the labeled dataset from
a Kaggle dataset (pushed by GitHub Actions), trains, and pushes the result
straight to HF Hub itself — gated on eval accuracy, same as push_artifacts.py.
"""
import os, numpy as np, pandas as pd, torch, joblib
import nlpaug.augmenter.word as naw
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

INPUT_CSV = "/kaggle/input/drug-reviews-labeled/drug_reviews_labeled.csv"
MODEL_OUTPUT_DIR = "/kaggle/working/model"
HF_MODEL_REPO = "yogeshagowda/mtech-model"
MIN_ACCURACY = 0.55

hf_token = UserSecretsClient().get_secret("HF_TOKEN")

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

def augment_data(df, target_per_cat=500):
    """FIXED: caps large categories, not just tops up small ones — otherwise a
    248k-row source dataset trains on all 248k rows every run."""
    aug = naw.SynonymAug(aug_src='wordnet')
    balanced = []
    for cat in df['review_category'].unique():
        cat_df = df[df['review_category'] == cat]
        n = len(cat_df)
        if n > target_per_cat:
            cat_df = cat_df.sample(n=target_per_cat, random_state=42)
        elif n < target_per_cat:
            needed = target_per_cat - n
            texts = [aug.augment(cat_df.iloc[i % n]['review_text'])[0] for i in range(needed)]
            cat_df = pd.concat([cat_df, pd.DataFrame({'review_text': texts, 'review_category': cat})])
        balanced.append(cat_df)
    return pd.concat(balanced).sample(frac=1).reset_index(drop=True)

def main():
    df = pd.read_csv(INPUT_CSV).dropna(subset=['review_text'])
    df = augment_data(df, target_per_cat=500)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['review_category'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review_text'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label'], random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)

    training_args = TrainingArguments(
        output_dir='/kaggle/working/results', num_train_epochs=4, per_device_train_batch_size=16,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="accuracy", fp16=torch.cuda.is_available(), report_to="none")

    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (np.argmax(p.predictions, axis=-1) == p.label_ids).mean()})

    trainer.train()
    eval_result = trainer.evaluate()
    acc = eval_result.get('eval_accuracy', 0)
    print(f"Eval accuracy: {acc:.4f}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    joblib.dump(label_encoder, f"{MODEL_OUTPUT_DIR}/label_encoder.pkl")

    if acc < MIN_ACCURACY:
        print(f"Accuracy {acc:.4f} below floor {MIN_ACCURACY} — NOT pushing to HF Hub")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=MODEL_OUTPUT_DIR, repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Pushed model to https://huggingface.co/{HF_MODEL_REPO}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
