"""Basic SD1.5 closed loop and shared orchestration for V3."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.pix2gestalt_prior import synthesize_amodal_prior
from src.vlm.clip_verify import verify_prompt_with_clip
from src.stages.appearance_prior import build_oracle_appearance_seed
from src.stages.inpaint_refiner import refine_exact_mask
from src.models.sd15_ip_adapter import validate_v2_config
from src.vlm.qwen_basic import (build_vlm_visual_prompts, run_basic_vlm_reasoning,
    run_target_aware_critique, truncate_tokens_clean, DEFAULT_NEGATIVE_PROMPT)
from src.vision.compositing import adaptive_seam_blend, finalize_rgba
from src.vision.roi import extract_square_roi, mask_bbox
from src.eval.metrics import compute_appearance_coherence_metrics


def critic_panels(image, visible, amodal, completed, size=384):
    """Four spatially registered panels; background and target cannot swap identities."""
    y1, x1, y2, x2 = mask_bbox(amodal)
    side = max(32, int(round(max(y2 - y1 + 1, x2 - x1 + 1) * 1.2)))
    roi = dict(y0=(y1 + y2) // 2 - side // 2, x0=(x1 + x2) // 2 - side // 2, size=side)
    observed = np.full_like(image, 255)
    observed[visible] = image[visible]
    result = np.full_like(image, 255)
    result[amodal] = completed[amodal]
    regions = np.zeros_like(image)
    regions[visible] = 255
    regions[amodal & ~visible] = [40, 130, 255]
    arrays = [image, observed, result, regions]
    return [np.array(Image.fromarray(extract_square_roi(a, roi, pad_value=0 if i == 3 else 255))
                     .resize((size, size), Image.Resampling.NEAREST if i == 3 else Image.Resampling.LANCZOS))
            for i, a in enumerate(arrays)]


def candidate_rank(critique):
    """A valid PASS always outranks a rejection; invalid JSON is not a score of zero."""
    score = critique.get("score")
    return (bool(critique.get("accepted")), float(score) if score is not None else float("-inf"))


class V2Pipeline:
    def _describe(self, context, isolated, query):
        return run_basic_vlm_reasoning(self.models.vlm, context, isolated, user_query=query)

    def __init__(self, models, config):
        validate_v2_config(config)
        self.models, self.config, self.device = models, config, models.device
    def _reason(self, image, vis, query):
        # Step 1 uses only image + visible mask; no hidden-mask geometry or labels.
        context, isolated, panel = build_vlm_visual_prompts(image, vis, vis, target_size=384)
        reasoning = self._describe(context, isolated, query)
        return reasoning, context, isolated, panel

    def _prior(self, image, vis, occ, job):
        amodal = vis | occ
        raw = synthesize_amodal_prior(self.models, self.config, image, vis, amodal, job)
        info = build_oracle_appearance_seed(image, vis, occ, raw["p2g_rgb_full"], raw["p2g_alpha_full"])
        return amodal, raw, info

    def _verify_prompt(self, isolated, reasoning, query):
        return verify_prompt_with_clip(self.models.clip_model, self.models.clip_processor,
                                       self.device, isolated, reasoning, user_query=query)

    def _inpaint_pipe(self, isolated, seed):
        return self.models.pipe

    def _refine(self, prior, vis, amodal, isolated, prompt, negative, strength, guidance, seed):
        # Requesting .pipe offloads Qwen before activating SD1.5.
        pipe = self._inpaint_pipe(isolated, seed)
        common = dict(pipe=pipe, device=self.device, seed_canvas=prior["seed_canvas"], visible_mask=vis,
                      prompt=prompt, negative_prompt=negative, strength=strength, guidance_scale=guidance,
                      num_inference_steps=int(self.config.get("num_inference_steps", 28)), seed=seed,
                      ref_p90_luma=float(prior["ref_p90_luma"]))
        rgb = refine_exact_mask(**common, occluded_mask=amodal & ~vis,
                                unfilled_mask=prior["unfilled_mask"])
        return rgb, getattr(pipe, "diagnostics", {})

    def _critique(self, panels, context, name):
        old = run_target_aware_critique(self.models.vlm, panels[2], context, name)
        accepted = old["score"] >= self.config["critique_threshold"]
        return dict(old, accepted=accepted, final_decision="PASS" if accepted else "REJECT",
                    reason=old["feedback"], repair_instruction=old["feedback"], valid_json=None)

    def complete(self, image, visible_mask, text_query="", job_dir=None, occluded_mask=None):
        t0 = time.perf_counter()
        vis = np.asarray(visible_mask, bool)
        if image.dtype != np.uint8 or image.shape != (*vis.shape, 3) or not vis.any():
            raise ValueError("V2 requires RGB uint8 and a nonempty matching visible mask")
        if occluded_mask is None or np.asarray(occluded_mask).shape != vis.shape:
            raise ValueError("A matching hidden mask is required")
        occ = np.asarray(occluded_mask, bool) & ~vis
        job = Path(job_dir) if job_dir is not None else None
        if job:
            job.mkdir(parents=True, exist_ok=True)
            (job / "critic_logs.jsonl").write_text("", encoding="utf-8")

        def save_json(name, data):
            if job:
                (job / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

        reasoning, context, isolated, panel = self._reason(image, vis, text_query)
        save_json("micro_attributes.json" if self.config["use_micro_attributes"] else "basic_reasoning.json", reasoning)
        if job:
            (job / "reasoning.txt").write_text(reasoning["raw"], encoding="utf-8")
            Image.fromarray(panel).save(job / "vlm_visual_prompt.jpg")
            Image.fromarray(isolated).save(job / "visible_reference.png")
        amodal, raw_prior, prior = self._prior(image, vis, occ, job)
        missing = amodal & ~vis
        pad = dict(needs_outpaint=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)
        save_json("outpainting.json", pad)
        if job:
            Image.fromarray(prior["seed_canvas"]).save(job / "prior_rgb.png")

        # Preserve CLIP verification as a guard on object identity. When Qwen is
        # accepted, retain its attributes even if CLIP prefers a shorter noun phrase.
        prompt, clip_info = self._verify_prompt(isolated, reasoning, text_query)
        if self.config["use_micro_attributes"] and clip_info["verified"] and reasoning.get("valid_json"):
            prompt = reasoning["sd_prompt"]
        save_json("prompt_selection.json", dict(prompt=prompt, clip_verification=clip_info))
        tokenizer = self.models.pipe.tokenizer
        base_prompt = truncate_tokens_clean(prompt, tokenizer, max_tokens=60)
        negative = truncate_tokens_clean(DEFAULT_NEGATIVE_PROMPT, tokenizer, max_tokens=70)
        fraction = float(prior["unfilled_fraction"])
        high_strength = self.config.get("unfilled_refine_strength", .62)
        strength = float(high_strength if fraction >= .25 else self.config.get("prior_refine_strength", .24))
        history, best, best_critique, selected = [], None, None, None
        limit = self.config["max_critic_iter"] if self.config.get("enable_critic", True) else 1
        for attempt in range(limit if missing.any() else 0):
            seed = int(self.config["seed"]) + attempt * 1009
            guidance = min(9.0, float(self.config["guidance_scale"]) + attempt * float(self.config["retry_guidance_step"]))
            repair = history[-1].get("repair_instruction", "") if history else ""
            # Keep attributes and leave room for a concise positive repair request.
            active_prompt = truncate_tokens_clean(
                (truncate_tokens_clean(base_prompt, tokenizer, 40) + ", " + repair) if repair else base_prompt,
                tokenizer, 70)
            start = time.perf_counter()
            generated, diagnostics = self._refine(prior, vis, amodal, isolated, active_prompt, negative,
                                                   strength, guidance, seed)
            blended = adaptive_seam_blend(image, generated, vis, amodal)
            # Final observed pixels are exactly preserved, independently of latent/VAE effects.
            blended[vis] = image[vis]
            rgba = finalize_rgba(blended, amodal, *image.shape[:2])
            synthesis_seconds = time.perf_counter() - start
            panels = critic_panels(image, vis, amodal, blended)
            if self.config.get("enable_critic", True):
                critique = self._critique(panels, context, reasoning["object_name"])
            else:
                critique = dict(score=None, accepted=False, final_decision="NOT_EVALUATED",
                                reason="Critic disabled for ablation", repair_instruction="", valid_json=None)
            entry = dict(critique, attempt=attempt + 1, retries=attempt, seed=seed, guidance_scale=guidance,
                         strength=strength, prompt=active_prompt, synthesis_seconds=round(synthesis_seconds, 3),
                         attempt_seconds=round(time.perf_counter() - start, 3), refinement=diagnostics)
            history.append(entry)
            if best is None or candidate_rank(critique) > candidate_rank(best_critique):
                best, best_critique, selected = rgba, critique, attempt + 1
            if job:
                Image.fromarray(rgba).save(job / f"candidate_{attempt + 1:02d}.png")
                Image.fromarray(np.concatenate(panels, axis=1)).save(job / f"critic_input_{attempt + 1:02d}.png")
                with (job / "critic_logs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                save_json("critique_history.json", history)
            print(f"[V{self.config['pipeline_version']} Critic {attempt + 1}/{limit}] {entry['final_decision']} | {entry['reason']}", flush=True)
            if critique["accepted"]:
                break

        if best is None:
            best = finalize_rgba(image, amodal, *image.shape[:2])
        accepted = bool(best_critique and best_critique["accepted"])
        outcome = "no_missing_region" if not history else (
            "not_evaluated" if not self.config.get("enable_critic", True) else "accepted" if accepted else "exhausted_retries")
        trace = dict(pad_info=pad, pipeline_version=self.config["pipeline_version"], synthesis_attempts=len(history),
                     retries=max(0, len(history) - 1), selected_attempt=selected, outcome=outcome,
                     critic_accepted=accepted, critique_history=history, shape_diagnostics=raw_prior["diagnostics"],
                     backend=f"V{self.config['pipeline_version']} mask-given",
                     seconds=round(time.perf_counter() - t0, 3),
                     coherence=compute_appearance_coherence_metrics(image, vis, missing, best[:, :, :3]))
        save_json(f"v{self.config['pipeline_version']}_selection.json", {k: trace[k] for k in
                  ("selected_attempt", "outcome", "critic_accepted", "synthesis_attempts", "retries")})
        return dict(input_image=image, visible_mask=vis, amodal_mask=amodal, inpainted_rgba=best,
                    vlm_reasoning=reasoning["raw"], clip_verification=clip_info, critique_history=history,
                    final_score=best_critique["score"] if best_critique else None,
                    critic_accepted=accepted, _trace=trace)
