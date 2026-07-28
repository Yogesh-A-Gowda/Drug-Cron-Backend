"""
Runs on Kaggle's infrastructure (GPU-enabled). Pulls the labeled dataset from
the attached Kaggle input path, trains, and saves the model locally 
so GitHub Actions can handle the final artifact deployment.
"""

import subprocess
import sys

# Install required packages dynamically in the Kaggle environment
subprocess.check_call([sys.executable, "-m", "pip", "install", "nlpaug", "transformers", "torch", "scikit-learn"])

# Now your standard imports will run smoothly
import nlpaug.augmenter.word as naw

import os, numpy as np, pandas as pd, torch, joblib
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

INPUT_CSV = "/kaggle/input/datasets/yogeshagowdaiiitdwd/drug-reviews-labeled/drug_reviews_labeled.csv"
MODEL_OUTPUT_DIR = "/kaggle/working/model"
MIN_ACCURACY = 0.55

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
    """Caps large categories, and tops up small ones using synonym augmentation."""
    aug = naw.SynonymAug(aug_src='wordnet')
    balanced = []
    for cat in df['review_category'].unique():
        cat_df = df[df['review_category'] == cat].copy()
        n = len(cat_df)
        if n > target_per_cat:
            cat_df = cat_df.sample(n=target_per_cat, random_state=42)
        elif n < target_per_cat:
            needed = target_per_cat - n
            texts = [aug.augment(cat_df.iloc[i % n]['review_text'])[0] for i in range(needed)]
            aug_df = pd.DataFrame({'review_text': texts, 'review_category': cat})
            cat_df = pd.concat([cat_df, aug_df], ignore_index=True)
        balanced.append(cat_df)
    return pd.concat(balanced).sample(frac=1).reset_index(drop=True)

def main():
    print(f"Loading dataset from attached input path: {INPUT_CSV}...")
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

    # Save outputs and evaluation metrics locally for GitHub Actions
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    joblib.dump(label_encoder, f"{MODEL_OUTPUT_DIR}/label_encoder.pkl")

    metrics_path = f"{MODEL_OUTPUT_DIR}/eval_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(str({"eval_accuracy": acc}))

    if acc < MIN_ACCURACY:
        print(f"Accuracy {acc:.4f} below floor {MIN_ACCURACY} — artifacts saved, but performance gate failed.")
    else:
        print(f"Accuracy {acc:.4f} met threshold. Model ready for deployment.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
