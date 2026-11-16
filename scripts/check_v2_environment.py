"""Validate V2 config and installed callback API before loading foundation weights."""
import argparse
import inspect
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.sd15_ip_adapter import validate_v2_config


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3_sd15_proposed.json")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--check-access", action="store_true", help="Check backbone/adapter configs and files without downloading large weights")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_v2_config(config)
    if not args.config_only:
        for name, expected in (("diffusers", "0.37.1"), ("transformers", "5.0.0")):
            actual = version(name)
            if actual != expected:
                raise RuntimeError(f"{name}={actual}, expected {expected}; rerun notebook cell 3 after pulling code")
        from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
        from transformers import Qwen3VLForConditionalGeneration
        keys = inspect.signature(StableDiffusionInpaintPipeline.__call__).parameters
        if not {"callback_on_step_end", "callback_on_step_end_tensor_inputs", "ip_adapter_image"} <= set(keys):
            raise RuntimeError("Missing required inpainting callback or image conditioning API")
        print("V2 runtime APIs and pinned versions OK (weights have not been loaded).")
    if args.check_access:
        from huggingface_hub import HfApi, hf_hub_download
        unet = json.loads(Path(hf_hub_download(config["inpainting_model_id"], "unet/config.json")).read_text())
        if unet.get("in_channels") != 9:
            raise ValueError("Checkpoint is not a 9-channel inpainting model")
        if config["use_ip_adapter"]:
            if unet.get("cross_attention_dim") != 768:
                raise ValueError("Incompatible SD1.5 adapter / backbone attention dimensions")
            repo = config["ip_adapter_model_id"]
            if not HfApi().file_exists(repo_id=repo, filename="models/" + config["ip_adapter_weight_name"]):
                raise ValueError("IP-Adapter Plus checkpoint is unavailable")
            encoder = json.loads(Path(hf_hub_download(repo, "models/image_encoder/config.json")).read_text())
            if encoder.get("hidden_size") != 1280:
                raise ValueError("IP-Adapter Plus SD1.5 requires the CLIP ViT-H patch encoder")
        print("Backbone / adapter access and architecture checks OK.")
    print("V2 configuration OK:", args.config)
    print("Backbone:", config["inpainting_model_id"], "| IP-Adapter:", config["use_ip_adapter"])
    print("Max attempts:", config["max_critic_iter"], "(includes the first attempt)")


if __name__ == "__main__":
    main()
