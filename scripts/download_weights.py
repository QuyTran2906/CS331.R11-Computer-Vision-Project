"""Fetch the author's official Pix2Gestalt checkpoint without global torch.load overrides."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    folder=ROOT/"ckpt"
    folder.mkdir(exist_ok=True)
    target=folder/"epoch=000005.ckpt"
    if not target.exists():
        if shutil.disk_usage(ROOT).free < 20*1024**3:
            raise RuntimeError("Need at least 20 GiB free disk for the large checkpoint alone")
        cached=hf_hub_download("cvlab/pix2gestalt-weights",filename="epoch=000005.ckpt")
        target.symlink_to(Path(cached).resolve())
    if target.stat().st_size<1024**3:
        raise RuntimeError("Checkpoint too small; may be incomplete or a Git LFS pointer")
    print("Pix2Gestalt checkpoint:",target.stat().st_size,"bytes")
    for repo, includes, excludes in (
        ("Qwen/Qwen3-VL-4B-Instruct", ["*.json", "*.txt", "*.jinja", "*.safetensors"], []),
        ("stable-diffusion-v1-5/stable-diffusion-inpainting",
         ["model_index.json", "unet/*", "vae/*", "text_encoder/*", "tokenizer/*",
          "scheduler/*", "feature_extractor/*"], ["*.fp16.*", "*.onnx", "*.xml", "*.msgpack"]),
        ("h94/IP-Adapter", ["models/ip-adapter-plus_sd15.safetensors",
                            "models/image_encoder/config.json", "models/image_encoder/model.safetensors"], []),
    ):
        print(f"Caching {repo} ...", flush=True)
        snapshot_download(repo, allow_patterns=includes, ignore_patterns=excludes, max_workers=2)
    print("Checkpoint cache ready. CLIP text verifier and CarveKit download on first use.")

if __name__=="__main__": main()
