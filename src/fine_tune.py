"""
Adapted from Mtech_BERT_Classifier.ipynb. Retrains from base distilbert-base-uncased
on the full accumulated + labeled dataset each run (simpler than incremental
fine-tuning, and cheap enough on CPU for this dataset size). Removes all
Colab-specific cells (drive.mount, files.upload/download).
"""
import argparse, os
import pandas as pd, numpy as np, torch, joblib
import nlpaug.augmenter.word as naw
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

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
    aug = naw.SynonymAug(aug_src='wordnet')
    balanced = []
    for cat in df['review_category'].unique():
        cat_df = df[df['review_category'] == cat]
        n = len(cat_df)
        if n < target_per_cat:
            needed = target_per_cat - n
            texts = [aug.augment(cat_df.iloc[i % n]['review_text'])[0] for i in range(needed)]
            cat_df = pd.concat([cat_df, pd.DataFrame({'review_text': texts, 'review_category': cat})])
        balanced.append(cat_df)
    return pd.concat(balanced).sample(frac=1).reset_index(drop=True)

def run(input_csv, model_output_dir):
    df = pd.read_csv(input_csv).dropna(subset=['review_text'])
    df = augment_data(df, target_per_cat=500)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['review_category'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review_text'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label'], random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        output_dir='./results', num_train_epochs=4, per_device_train_batch_size=16,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="accuracy", fp16=torch.cuda.is_available(), report_to="none")

    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (np.argmax(p.predictions, axis=-1) == p.label_ids).mean()})

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Eval accuracy: {eval_result.get('eval_accuracy')}")

    os.makedirs(model_output_dir, exist_ok=True)
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    joblib.dump(label_encoder, f"{model_output_dir}/label_encoder.pkl")

    with open(f"{model_output_dir}/eval_metrics.txt", "w") as f:
        f.write(str(eval_result))

    print(f"Model saved to {model_output_dir}")
    return eval_result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(args.input, args.output_dir)
