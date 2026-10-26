"""Writes kaggle_job/kernel-metadata.json for the Kaggle account configured on this computer.
Only the Kaggle *username* is read; the API key is never printed or copied."""
import json
import os
import pathlib
import subprocess
import sys

SLUG = "spend-categorization-llama3-qlora"
here = pathlib.Path(__file__).resolve().parent
user = os.environ.get("KAGGLE_USERNAME")
token_file = pathlib.Path.home() / ".kaggle" / "kaggle.json"
if not user and token_file.exists():
    try:
        user = json.loads(token_file.read_text()).get("username")
    except Exception:
        pass
if not user:
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            if line.strip().lower().startswith("- username") or line.strip().lower().startswith("username"):
                user = line.split(":", 1)[1].strip()
    except Exception:
        pass
if not user:
    print("NO_KAGGLE_ACCOUNT: create an API token at https://www.kaggle.com/settings (API -> Create New Token)")
    print(f"and save the downloaded kaggle.json to {token_file}")
    sys.exit(2)
meta = {
    "id": f"{user}/{SLUG}",
    "title": "Spend Categorization Llama3 QLoRA",
    "code_file": "train_kaggle.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(here / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
(here / "kaggle_user.txt").write_text(user)
print(f"KAGGLE_USER_OK: kernel id = {user}/{SLUG}")
