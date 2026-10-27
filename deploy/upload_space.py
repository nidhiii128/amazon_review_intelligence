"""Create (or update) the public Hugging Face Space and upload the bundle.

    python deploy/upload_space.py <space-name>

Needs a Hugging Face token with write access, either logged in once with
`hf auth login` or exported as HF_TOKEN. Run deploy/build_space.py first.
"""
import pathlib
import sys

from huggingface_hub import HfApi

bundle = pathlib.Path(__file__).resolve().parent / "hf_space"
name = sys.argv[1] if len(sys.argv) > 1 else "reviewiq"
api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{name}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
api.upload_folder(folder_path=str(bundle), repo_id=repo_id, repo_type="space",
                  commit_message="Deploy ReviewIQ dashboard")
print(f"Uploaded. Build logs and app: https://huggingface.co/spaces/{repo_id}")
print(f"Public app URL: https://{user.lower()}-{name.lower()}.hf.space")
