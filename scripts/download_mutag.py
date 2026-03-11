from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "graphs-datasets/MUTAG"

repo_root = Path(__file__).resolve().parent.parent
local_dir = repo_root / "data" / "raw" / "MUTAG"

# download the dataset repo
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(local_dir),
)

print(f"Dataset downloaded to {local_dir}")