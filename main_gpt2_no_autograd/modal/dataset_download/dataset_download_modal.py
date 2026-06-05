import modal
import os
from pathlib import Path

# Match this to the volume name used in your training script!
VOLUME_NAME = "nanogpt-no-cupy-autograd-runs"
volume = modal.Volume.from_name(VOLUME_NAME)

app = modal.App("hydrate-openwebtext")
image = modal.Image.debian_slim().pip_install("kaggle")

@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    timeout=60 * 60, # 1 hour timeout for the large download/unzip
)
def download_kaggle_data():
    import kaggle

    # This matches the DATA_DIR path we set up in your training script
    out_dir = Path("/checkpoints/data/openwebtext")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and extracting florian0627/openwebtext to {out_dir}...")
    
    # The Kaggle API handles the download and the unzipping automatically
    kaggle.api.dataset_download_cli(
        "florian0627/openwebtext", 
        path=str(out_dir), 
        unzip=True
    )

    print("Commiting files to the persistent volume...")
    volume.commit()
    print("Done! The .bin files are now permanently stored in your Modal volume.")

@app.local_entrypoint()
def main():
    download_kaggle_data.remote()