from huggingface_hub import HfApi, create_repo
import os
import sys

def push_to_hf(repo_id, token):
    api = HfApi()
    
    print(f"Creating repository {repo_id}...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print("Uploading model...")
    api.upload_file(
        path_or_fileobj="tomato_leaf_sklearn_model.joblib",
        path_in_repo="tomato_leaf_sklearn_model.joblib",
        repo_id=repo_id,
        token=token
    )
    
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj="hf_model_card.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token
    )
    
    print(f"Successfully pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python push_to_hf.py <username/repo-name> <your-hf-token>")
    else:
        push_to_hf(sys.argv[1], sys.argv[2])
