"""Entry point for the three SD1.5 experiments."""
from pathlib import Path

from PIL import Image

from src.models.memory_manager import FoundationModels
from src.models.sd15_ip_adapter import validate_v2_config
from src.pipelines.v1_direct import V1Direct, validate_sd15_direct


def validate_experiment_config(config):
    version = config.get("pipeline_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Choose pipeline version 1, 2 or 3")
    if config.get("use_occluded_mask_input") is not True:
        raise ValueError("All versions require the given hidden mask")
    if config.get("text_query_policy", "empty") != "empty":
        raise ValueError("Ground-truth category prompts are disabled")
    if version == 1:
        if config.get("pipeline_mode") != "sd15_direct":
            raise ValueError("V1 requires sd15_direct")
        validate_sd15_direct(config)
    else:
        validate_v2_config(config)
        flags = ("use_micro_attributes", "use_latent_blending", "use_strict_critic")
        if any(config[key] != (version == 3) for key in flags):
            raise ValueError("Micro-attributes, latent blending and strict critic belong to V3")
        if not config["enable_critic"] or (version == 2 and config["use_ip_adapter"]):
            raise ValueError("V2/V3 require critic feedback; the adapter belongs to V3")
    return version


class AmodalBackend:
    def __init__(self, config):
        import torch

        version = validate_experiment_config(config)
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This configuration requires CUDA. Use the Colab notebook.")
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        self.models = FoundationModels(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config.get("clip_model_id", "openai/clip-vit-base-patch32"),
            vlm_model_id=config.get("vlm_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device="cuda",
            advanced_config=config if version != 1 else None,
            refinement_scheduler="ddim",
        )
        self.job = None
        if version == 1:
            self.completer = V1Direct(self.models, config)
        elif version == 2:
            from src.pipelines.v2_pipeline import V2Pipeline
            self.completer = V2Pipeline(self.models, config)
        else:
            from src.pipelines.v3_proposed import V3Proposed
            self.completer = V3Proposed(self.models, config)

    def begin_sample(self, job):
        self.job = Path(job)

    def complete(self, image, visible, label, occluded_mask=None):
        try:
            return self.completer.complete(image=image, visible_mask=visible,
                                           occluded_mask=occluded_mask, job_dir=self.job)
        finally:
            self.models.release_gpu()
