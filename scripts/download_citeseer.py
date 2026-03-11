from pathlib import Path
from urllib.request import urlretrieve
import tarfile

archive_url = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"

repo_root = Path(__file__).resolve().parent.parent
local_dir = repo_root / "data" / "raw" / "citeseer"
archive_path = local_dir / "citeseer.tgz"

local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {archive_url}...")
urlretrieve(archive_url, archive_path)

print(f"Extracting {archive_path}...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=local_dir, filter="data")

archive_path.unlink(missing_ok=True)

print(f"CiteSeer raw files downloaded to {local_dir / 'citeseer'}")