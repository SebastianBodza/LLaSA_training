import modal 
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    dataset_name: str = "SebastianBodza/Kartoffelphon"
    output_dataset_name: str = "SebastianBodza/Kartoffelphon-processed"
    output_dir: str = "/data/kartoffelphon_full"
    debug: bool = False
    max_length: int = 4096
    xcodec_model: str = "HKUSTAudio/xcodec2"


app = modal.App(name="llasa-german-processing")

# Creating the volume
DATA_DIR = "/data"
volume = modal.Volume.from_name("llasa-german", create_if_missing=True)

# Creating the volume
cuda_version = "12.5.1"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

image = (
    cuda_dev_image.apt_install(
        "ffmpeg",
        "build-essential",  # Adds gcc, g++, make
        "clang",  # Specifically add clang
        "python3-dev",  # Python development headers
    )
    .pip_install(
        "accelerate==1.1.0",
        "bitsandbytes==0.45.1",
        "datasets==3.0.1",
        "deepspeed==0.15.1",
        "einops==0.8.0",
        "hf_transfer==0.1.9",
        "huggingface-hub==0.25.1",
        "librosa==0.10.2.post1",
        "liger_kernel==0.5.2",
        "lightning-utilities==0.11.7",
        "multiprocess==0.70.16",
        "numba==0.60.0",
        "numpy==2.0.2",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvrtc-cu12==12.1.105",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106",
        "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106",
        "nvidia-ml-py==12.560.30",
        "nvidia-nccl-cu12==2.20.5",
        "nvidia-nvjitlink-cu12==12.6.68",
        "nvidia-nvtx-cu12==12.1.105",
        "pandas==2.2.3",
        "psutil==6.0.0",
        "pyarrow==17.0.0",
        "pydantic==2.9.2",
        "pydantic_core==2.23.4",
        "pydub==0.25.1",
        "python-dotenv==1.0.1",
        "pytorch-lightning==2.4.0",
        "safetensors==0.4.5",
        "scipy==1.13.1",
        "sentencepiece==0.2.0",
        "soundfile==0.12.1",
        "sympy==1.13.3",
        "threadpoolctl==3.5.0",
        "tiktoken==0.8.0",
        "tokenizers",
        "torch==2.4.1",
        "torchao==0.5.0",
        "torchaudio==2.4.1",
        "torchmetrics==1.4.2",
        "torchtune==0.3.1",
        "tqdm==4.66.5",
        "transformers",
        "triton==3.0.0",
        "vector-quantize-pytorch==1.17.8",
        "wandb==0.18.1",
        "xcodec2==0.1.4",
    )
    .env({"HF_HUB_CACHE_DIR": f"{DATA_DIR}/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Adding the HF_TOKEN TODO: Add this to the Modal secrets
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)



@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,  # min
)
def download_data(config):
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config.dataset_name,
        repo_type="dataset",
        local_dir=os.path.join(DATA_DIR, "dataset"),
        token=os.environ["HF_TOKEN"],
    )



@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,  # min
)
def download_model(config):
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config.xcodec_model,
        repo_type="model",
        local_dir=os.path.join(DATA_DIR, "model"),
        token=os.environ["HF_TOKEN"],
    )

@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="L40s:2",
    timeout=60 * 60 * 23,  # min
    mounts=[
        # Mount processing scripts
        modal.Mount.from_local_file(
            Path(__file__).parent / "original_translate_to_dataset_o3.py",
            remote_path="/root/original_translate_to_dataset_o3.py",
        ),
        modal.Mount.from_local_dir(
            local_path=Path(__file__).parent / "vq",
            remote_path="/root/vq",
        )
    ],
)
def process_audio(config):
    import subprocess
    import sys
    import os

    # Construct dataset path from downloaded location
    dataset_path = os.path.join(DATA_DIR, "dataset")

    # Build command arguments
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "original_translate_to_dataset_o3.py",
        "--dataset-dir",
        dataset_path,
        "--ckpt",
        os.path.join(DATA_DIR, "model/ckpt/epoch=4-step=1400000.ckpt"),
        "--output-dir",
        config.output_dir,
        "--cache_dir",
        os.path.join(DATA_DIR, "cache"),
    ]

    # Add debug flag if needed
    if config.debug:
        command.append("--debug")

    # Run processing script with arguments
    subprocess.run(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
    volume.commit()


@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,  # min
)
def upload_large_folder(config):
    import os
    from huggingface_hub import HfApi

    api = HfApi()

    api.upload_large_folder(
        repo_id=config.output_dataset_name,
        repo_type="dataset",
        folder_path=config.output_dir,
        token=os.environ["HF_TOKEN"],
    )

@app.local_entrypoint()
def run():
    print("🧙‍♂️ loading data")
    # download_data.remote(SharedConfig())

    print("🧙‍♂️ loading model")
    # download_model.remote(SharedConfig())

    print("🧙‍♂️ processing audio")
    process_audio.remote(SharedConfig())

    upload_large_folder.remote(SharedConfig())

    print("✅ done")


# Run with modal run modal/run_on_modal.py 