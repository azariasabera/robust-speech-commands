# src/noise/download_dataset.py

import urllib.request
import tarfile
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
from .utils import get_noise_param


def download_musan(config: DictConfig) -> None:
    """
    Download the MUSAN dataset and extract only the noise/free-sound subset.

    Resulting structure:
        dir/
        ├─ musan.tar.gz
        └─ musan/
           └─ noise/
              └─ free-sound/
                 └─ *.wav

    Args:
        config: Hydra DictConfig object containing noise settings.
    """
    url: str = get_noise_param(config, key="url", default="http://www.openslr.org/resources/17/musan.tar.gz")
    root_dir = Path(get_noise_param(config, key="dir", default=str(Path.cwd()/"data"/"noise"))).expanduser().resolve()

    root_dir.mkdir(parents=True, exist_ok=True)

    tar_path = root_dir / "musan.tar.gz"

    # Download archive
    if not tar_path.exists():
        print(f"Downloading MUSAN from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print(f"MUSAN archive already exists: {tar_path}")

    # Extract ONLY musan/noise/free-sound
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith("musan/noise/free-sound/")
        ]
        for member in tqdm(members, desc="Extracting free-sound"):
            
            out_path = root_dir / member.name

            # Skip if already extracted
            if out_path.exists():
                continue
            tar.extract(member=member, path=root_dir)

    print("Extracted musan/noise/free-sound subset.")
