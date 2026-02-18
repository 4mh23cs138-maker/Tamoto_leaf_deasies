import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="e.g. username/tomato-leaf-disease-resnet18")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = __import__("os").environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACE_TOKEN before running upload_hf.py")

    artifact_dir = Path(args.artifact_dir)
    required = ["best_model.pt", "metrics.json", "classification_report.json"]

    for name in required:
        if not (artifact_dir / name).exists():
            raise FileNotFoundError(f"Missing artifact: {artifact_dir / name}")

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)
    api = HfApi(token=token)

    for name in required:
        upload_file(
            path_or_fileobj=str(artifact_dir / name),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
        )

    readme = artifact_dir / "README_model.md"
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))

    model_card = f"""# Tomato Leaf Disease Classifier (ResNet18)\n\nBest validation accuracy: {metrics.get('best_val_acc', 'n/a')}\n\n## Labels\n{', '.join(metrics.get('classes', []))}\n\n## Training\n- Epochs: {metrics.get('epochs')}\n- Batch size: {metrics.get('batch_size')}\n- LR: {metrics.get('lr')}\n- Image size: {metrics.get('img_size')}\n"""

    readme.write_text(model_card, encoding="utf-8")
    upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
    )

    print(f"Uploaded model artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
