"""Single-pass inpainting with a fixed prompt and pixel-exact visible preservation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.sd15_ip_adapter import SD15_INPAINTING
from src.stages.inpaint_refiner import run_inpaint_roi_pass
from src.vlm.qwen_basic import DEFAULT_NEGATIVE_PROMPT
from src.vision.compositing import adaptive_seam_blend, finalize_rgba
from src.vision.image_ops import binary_dilate, binary_erode
from src.vision.roi import mask_bbox

DEFAULT_PROMPT = (
    "A photograph of a single complete object on a plain white background, "
    "natural colors, realistic texture."
)


def prepare_direct_inputs(image, visible_mask, occluded_mask):
    """Same ROI/mask geometry as oracle refinement, without any appearance prior."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Direct inpainting requires uint8 HxWx3 RGB")
    if vis.shape != image.shape[:2] or occ.shape != vis.shape or not vis.any():
        raise ValueError("Invalid/empty visible mask or mismatched hidden mask")
    if (vis & occ).any():
        raise ValueError("Visible and hidden masks must be disjoint")
    amodal = vis | occ
    # Never expose occluder pixels to the baseline, even when strength < 1.
    canvas = np.full_like(image, 255)
    canvas[vis] = image[vis]
    canvas[occ] = 127
    y1, x1, y2, x2 = mask_bbox(amodal)
    side = max(128, int(round(max(y2 - y1 + 1, x2 - x1 + 1) * 1.28)))
    roi = dict(y0=((y1 + y2) // 2) - side // 2,
               x0=((x1 + x2) // 2) - side // 2, size=side, is_full_frame=False)
    inpaint_region = binary_dilate(occ, 2) & binary_dilate(amodal, 2) & ~binary_erode(vis, 2)
    if not inpaint_region.any():
        inpaint_region = occ.copy()
    return canvas, vis, occ, amodal, roi, inpaint_region


class DirectInpaintCompleter:
    baseline_name = "Direct inpainting"
    settings_filename = "inpaint_settings.json"
    timing_key = "inpaint_seconds"

    def __init__(self, models, config):
        self.models, self.config = models, config
        if config.get("text_query_policy", "empty") != "empty" or config.get("enable_critic", False):
            raise ValueError(f"{self.baseline_name} uses a fixed prompt and no critic")
        self.prompt = config.get("prompt", DEFAULT_PROMPT)
        self.strength = float(config.get("strength", 1.0))
        self.guidance = float(config.get("guidance_scale", 7.0))
        self.steps = int(config.get("num_inference_steps", 28))
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("A fixed nonempty prompt is required")
        if not 0.16 <= self.strength <= 1 or self.steps < 1 or self.guidance < 0:
            raise ValueError("Invalid direct inpainting settings")

    def complete(self, image, visible_mask, occluded_mask, job_dir=None):
        if occluded_mask is None:
            raise ValueError(f"{self.baseline_name} requires an input hidden mask")
        started = time.perf_counter()
        canvas, vis, occ, amodal, roi, inpaint_region = prepare_direct_inputs(image, visible_mask, occluded_mask)
        negative = self.config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        seed = int(self.config.get("seed", 42))
        settings = dict(prompt=self.prompt, negative_prompt=negative, seed=seed,
                        strength=self.strength, guidance_scale=self.guidance,
                        num_inference_steps=self.steps, resolution=512, roi=roi,
                        category_annotation_used=False, appearance_prior=False)
        if job_dir is not None:
            job = Path(job_dir)
            Image.fromarray(canvas).save(job / "baseline_input.png")
            Image.fromarray(inpaint_region.astype(np.uint8) * 255).save(job / "inpainting_mask.png")
            (job / "prompt_01.txt").write_text(self.prompt, encoding="utf-8")
            (job / self.settings_filename).write_text(json.dumps(settings, indent=2), encoding="utf-8")
        model_load_seconds, inference_seconds = 0.0, 0.0
        if occ.any():
            load_started = time.perf_counter()
            pipe = self.models.pipe  # The only model this baseline activates.
            model_load_seconds = time.perf_counter() - load_started
            print(f"[{self.baseline_name}] One pass, ROI={roi['size']}, steps={self.steps}, strength={self.strength}", flush=True)
            inference_started = time.perf_counter()
            generated = run_inpaint_roi_pass(
                pipe=pipe, device=self.models.device, seed_canvas=canvas,
                inpaint_region=inpaint_region, roi=roi, prompt=self.prompt,
                negative_prompt=negative, strength=self.strength, guidance_scale=self.guidance,
                num_inference_steps=self.steps, seed=seed,
            )
            inference_seconds = time.perf_counter() - inference_started
            # Match oracle compositing: only copy synthesized hidden pixels into
            # the seed, then call the shared seam/alpha code. No frequency locking.
            completed = canvas.copy()
            completed[occ] = generated[occ]
        else:
            completed = canvas.copy()
        rgb = adaptive_seam_blend(image, completed, vis, amodal)
        rgba = finalize_rgba(rgb, amodal, *image.shape[:2])
        if job_dir is not None:
            Image.fromarray(completed).save(job / "inpainting_01.png")
        return dict(inpainted_rgba=rgba, amodal_mask=amodal, final_score=None, critique_history=[],
                    _trace=dict(backend=f"{self.baseline_name} (fixed prompt, input hidden mask)", pad_info={},
                                synthesis_attempts=int(occ.any()), settings=settings,
                                model_load_seconds=round(model_load_seconds, 3),
                                **{self.timing_key: round(inference_seconds, 3)},
                                total_seconds=round(time.perf_counter() - started, 3)))


def validate_sd15_direct(config):
    if config.get("inpainting_model_id") != SD15_INPAINTING or config.get("refinement_scheduler") != "ddim":
        raise ValueError("SD1.5 Direct requires the SD1.5 inpainting checkpoint and DDIM, matching V2")
    if config.get("enable_critic") or config.get("text_query_policy", "empty") != "empty":
        raise ValueError("SD1.5 Direct requires a fixed category-free prompt and no critic")
    if any(config.get(key) for key in ("use_micro_attributes", "use_ip_adapter", "use_latent_blending")):
        raise ValueError("SD1.5 Direct cannot enable extra conditioning features")


class SD15DirectCompleter(DirectInpaintCompleter):
    baseline_name = "SD1.5 Direct"
    settings_filename = "sd15_settings.json"
    timing_key = "sd15_seconds"

    def __init__(self, models, config):
        validate_sd15_direct(config)
        super().__init__(models, config)

    def complete(self, *args, **kwargs):
        result = super().complete(*args, **kwargs)
        result["_trace"]["refinement_scheduler"] = "DDIMScheduler"
        return result


class V1Direct(SD15DirectCompleter):
    def complete(self, image, visible_mask, occluded_mask, job_dir=None):
        result = super().complete(image, visible_mask, occluded_mask, job_dir=job_dir)
        visible = np.asarray(visible_mask, dtype=bool)
        result["inpainted_rgba"][visible, :3] = image[visible]
        result["_trace"]["pipeline_version"] = 1
        return result
