"""Same direct-inpainting algorithm, explicitly paired with the SD1.5 V2 backbone."""
from src.pipelines.sd2_direct_completer import SD2DirectCompleter
from src.stages.sd2_advanced import SD15_INPAINTING


def validate_sd15_direct(config):
    if config.get("inpainting_model_id") != SD15_INPAINTING or config.get("refinement_scheduler") != "ddim":
        raise ValueError("SD1.5 Direct requires the SD1.5 inpainting checkpoint and DDIM, matching V2")
    if config.get("enable_critic") or config.get("text_query_policy", "empty") != "empty":
        raise ValueError("SD1.5 Direct requires a fixed category-free prompt and no critic")
    if any(config.get(key) for key in ("use_micro_attributes", "use_ip_adapter", "use_latent_blending")):
        raise ValueError("SD1.5 Direct cannot enable V2 conditioning features")


class SD15DirectCompleter(SD2DirectCompleter):
    baseline_name = "SD1.5 Direct"
    settings_filename = "sd15_settings.json"

    def __init__(self, models, config):
        validate_sd15_direct(config)
        super().__init__(models, config)

    def complete(self, *args, **kwargs):
        result = super().complete(*args, **kwargs)
        result["_trace"]["sd15_seconds"] = result["_trace"].pop("sd2_seconds")
        result["_trace"]["refinement_scheduler"] = "DDIMScheduler"
        return result
