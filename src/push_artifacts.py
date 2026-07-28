"""
Pushes the fine-tuned model to HF Hub, gated on an eval accuracy floor
so a bad scrape/run can't silently degrade the live serving model.
"""
import argparse, ast, os
from huggingface_hub import HfApi

MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.55"))

def run(model_dir, repo_id):
    metrics_path = f"{model_dir}/eval_metrics.txt"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = ast.literal_eval(f.read())
        acc = metrics.get("eval_accuracy", 0)
        if acc < MIN_ACCURACY:
            print(f"Eval accuracy {acc:.3f} below floor {MIN_ACCURACY} — NOT pushing this version")
            return False

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model")
    print(f"Pushed model to https://huggingface.co/{repo_id}")
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--repo_id", required=True)
    args = p.parse_args()
    run(args.model_dir, args.repo_id)
