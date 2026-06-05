import os
from pathlib import Path

import modal

APP_NAME = "nanoGPT-no-autograd"
VOLUME_NAME = "nanogpt-no-cupy-autograd-runs"
REMOTE_ROOT = Path("/app")

# I updated this path slightly to match the default folder name in your new script
REMOTE_OUT_DIR = Path("/checkpoints/out-gpt-ffn-bp-v3-cupy-openwebtext")

CPU_CORES = 16
MEMORY_MIB = 64 * 1024
TIMEOUT_SECONDS = 24 * 60 * 60

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy", "matplotlib", "cupy-cuda12x", "wandb", "tqdm", "datasets", "tiktoken")
    .env({
        # Forces Hugging Face to save the 54GB download to your persistent volume
        "HF_DATASETS_CACHE": "/checkpoints/hf_cache", 
        
        # Tells our scripts where to save and load the processed .bin files
        "DATA_DIR": "/checkpoints/data/openwebtext"   
    })
    .workdir(REMOTE_ROOT.as_posix())
    .add_local_dir(
        ".",
        REMOTE_ROOT.as_posix(),
        ignore=[
            ".git",
            ".vscode",
            "__pycache__",
            "*.pyc",
            "out-*",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=CPU_CORES,
    gpu="A100-80GB",
    memory=MEMORY_MIB,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT_SECONDS,
    volumes={"/checkpoints": volume},
)
def train():

    import no_autograd.train_gpt_ffn_bp_v3_cupy as trainer

    REMOTE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.out_dir = str(REMOTE_OUT_DIR)
    trainer.main()
    
    # This commit will save the checkpoints AND your new training_curves.png
    volume.commit()

    return f"Training finished. Checkpoints and plots are in Modal Volume {VOLUME_NAME}:{REMOTE_OUT_DIR}"


@app.local_entrypoint()
def main():
    print(train.spawn())