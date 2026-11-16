# Phạm vi: V2 dùng mô tả và critic dạng văn bản; V3 tái dùng crop đầu vào và giới hạn token.
"""Visual prompt rendering, Qwen3-VL reasoning/critique, and CLIP prompt verification."""
from __future__ import annotations

import re
import numpy as np
from PIL import Image

from src.vision.image_ops import binary_erode, mask_outline, resize_array
from src.vision.roi import compute_square_roi, extract_square_roi, mask_bbox

DEFAULT_NEGATIVE_PROMPT = (
    "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
    "blurry, low quality, distorted, deformed, extra limbs, floating head, "
    "text, watermark, artifacts"
)

# Giới hạn token theo tokenizer diffusion, tránh cắt theo số ký tự.
def truncate_tokens_clean(text: str, tokenizer, max_tokens: int = 70) -> str:
    """Truncate text to `max_tokens` without triggering HF `> 77` sequence warnings."""
    if not text:
        return ""
    try:
        ids = tokenizer(
            text,
            truncation=True,
            max_length=int(max_tokens),
            add_special_tokens=False,
        ).input_ids
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        words = text.split()
        return " ".join(words[:max_tokens])


# Tách tên vật thể, phần khuất và prompt từ các dòng có nhãn.
def parse_structured_vlm_reasoning(raw_text: str, fallback_query: str = "") -> dict[str, str]:
    """Extract `object_name`, `occluded_parts`, and clean diffusion `sd_prompt`."""
    cleaned = re.sub(r"\*+", "", raw_text or "").strip()

    obj_match = re.search(r"OBJECT\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    occ_match = re.search(r"OCCLUDED(?:_PARTS|\s+PARTS)?\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    prm_match = re.search(r"PROMPT\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)", cleaned, re.IGNORECASE | re.DOTALL)

    object_name = obj_match.group(1).strip(" .,\"'") if obj_match else ""
    occluded_parts = occ_match.group(1).strip() if occ_match else ""
    sd_prompt = " ".join(prm_match.group(1).split()).strip(" .,\"'") if prm_match else ""

    if not object_name:
        object_name = fallback_query.strip() or "object"

    if not sd_prompt:
        prose = " ".join(cleaned.split())
        prose = re.sub(
            r"^(the target object is|this is|inpainting prompt\s*:)\s*",
            "",
            prose,
            flags=re.IGNORECASE,
        )
        sd_prompt = prose.split(".")[0].strip() or f"complete {object_name} isolated on a white background"

    return {
        "object_name": object_name,
        "occluded_parts": occluded_parts,
        "sd_prompt": sd_prompt,
        "raw": raw_text.strip() if raw_text else "",
    }


# Làm tối bối cảnh và đánh dấu vật thể; chỉ tô vùng khuất khi caller gửi mask tương ứng.
def _render_censored_scene(image: np.ndarray, vis: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Dim background to 35%, censor `missing` with opaque slate stripes, and outline regions."""
    highlighted = (image.astype(np.float32) * 0.35).astype(np.uint8)
    if missing.any():
        yy, xx = np.indices(vis.shape)
        stripe = ((yy + xx) // 8) % 2 == 0
        censor = np.where(
            stripe[:, :, None],
            np.array([112, 116, 122], dtype=np.uint8),
            np.array([138, 142, 148], dtype=np.uint8),
        )
        highlighted[missing] = censor[missing]
    highlighted[vis] = image[vis]
    highlighted[mask_outline(missing, thickness=2)] = np.array([255, 215, 0], dtype=np.uint8)
    highlighted[mask_outline(vis, thickness=2)] = np.array([0, 255, 64], dtype=np.uint8)
    return highlighted


# Cô lập phần visible trên nền trắng làm tham chiếu màu và bề mặt.
def _render_isolated_visible_crop(image: np.ndarray, vis: np.ndarray, target_size: int) -> np.ndarray:
    vy1, vx1, vy2, vx2 = mask_bbox(vis)
    vh, vw = vy2 - vy1 + 1, vx2 - vx1 + 1
    iso_side = max(64, int(round(max(vh, vw) * 1.22)))
    iso_roi = {
        "y0": ((vy1 + vy2) // 2) - iso_side // 2,
        "x0": ((vx1 + vx2) // 2) - iso_side // 2,
        "size": iso_side,
        "is_full_frame": False,
    }
    white_scene = np.full_like(image, 255, dtype=np.uint8)
    white_scene[vis] = image[vis]
    iso_patch = extract_square_roi(white_scene, iso_roi, pad_value=255)
    return resize_array(iso_patch, (target_size, target_size), mode="linear")


# Tạo crop ngữ cảnh và crop visible cho V2/V3 cùng kích thước.
def build_vlm_visual_prompts(
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    target_size: int = 384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zoomed visual prompts (blind mode) with foreground occluders censored."""
    vis = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    missing = amodal & (~vis)

    highlighted = _render_censored_scene(image, vis, missing)
    ctx_roi = compute_square_roi(amodal, image.shape, context_scale=2.2, min_crop=192, full_frame_threshold=0.92)
    ctx_patch = extract_square_roi(highlighted, ctx_roi, pad_value=48)
    context_rgb = resize_array(ctx_patch, (target_size, target_size), mode="linear")

    isolated_rgb = _render_isolated_visible_crop(image, vis, target_size)

    sep = np.full((target_size, 8, 3), 220, dtype=np.uint8)
    panel_rgb = np.concatenate([context_rgb, sep, isolated_rgb], axis=1)
    return context_rgb, isolated_rgb, panel_rgb


# Chuẩn bị hai ảnh và chỉ giải mã token mới do Qwen sinh.
def _generate_qwen_vl(vlm, image_a: np.ndarray, image_b: np.ndarray, instruction: str, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    if vlm is None or vlm.model is None:
        raise RuntimeError("Strict mode: Qwen3-VL model is not loaded.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(image_a)},
                {"type": "image", "image": Image.fromarray(image_b)},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)

    with torch.no_grad():
        generated_ids = vlm.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
        )

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return vlm.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# Lấy mô tả cơ bản của V2 theo định dạng văn bản có nhãn.
def run_basic_vlm_reasoning(
    vlm,
    context_rgb: np.ndarray,
    isolated_rgb: np.ndarray,
    user_query: str = "",
) -> dict[str, str]:
    hint_line = f"Optional category hint: '{user_query.strip()}'.\n" if user_query and user_query.strip() else ""
    instruction = (
        "You are analyzing a partially occluded target object for amodal appearance completion.\n"
        "- Image 1 shows the local scene where ONLY the target object's visible region is at full "
        "brightness inside a BRIGHT GREEN outline, and the occluded region to reconstruct is covered "
        "by a DIAGONAL GRAY STRIPED CENSOR MASK inside a YELLOW outline.\n"
        "- Image 2 shows ONLY the target object's visible pixels isolated on a pure white background.\n"
        f"{hint_line}"
        "CRITICAL RULES:\n"
        "1. Identify ONLY the BRIGHT GREEN target object shown in Image 2.\n"
        "2. Write PROMPT as a concise (12-26 words) studio photo description of the COMPLETE, "
        "UNOCCLUDED target object isolated on a plain white background, matching the exact colors, "
        "materials, patterns, and lighting visible in Image 2.\n"
        "3. If the target object is a container, takeout box, tray, bowl, plate, dish, or surface "
        "whose interior was covered by the yellow-outlined censor mask, describe it as EMPTY and clean "
        "inside (NEVER imagine food, sandwiches, or items inside it).\n"
        "4. Do NOT mention occluders, masks, stripes, green/yellow outlines, or missing parts in PROMPT.\n\n"
        "Respond in this exact 3-line format:\n"
        "OBJECT: <short noun phrase naming the target object>\n"
        "OCCLUDED_PARTS: <one short sentence naming which part of the target object is hidden>\n"
        "PROMPT: <concise description of the complete unoccluded target object on a white background>"
    )
    raw_output = _generate_qwen_vl(vlm, context_rgb, isolated_rgb, instruction, max_new_tokens=140)
    return parse_structured_vlm_reasoning(raw_output, fallback_query=user_query)


# Đọc các điểm thành phần, tính điểm tổng hợp và lấy phản hồi chỉnh sửa.
def _parse_critic_response(output_text: str) -> dict:
    cleaned = re.sub(r"\*+", "", output_text)

    # Đọc điểm của tiêu chí cụ thể và giới hạn vào miền điểm hợp lệ.
    def _parse_score(label: str) -> float:
        match = re.search(rf"{label}\s*:\s*([\d.]+)", cleaned, re.IGNORECASE)
        if not match:
            raise ValueError(f"Critic response missing {label}: {output_text}")
        val = float(match.group(1))
        if not 0.0 <= val <= 10.0:
            raise ValueError(f"Critic score out of range: {label}={val}")
        return val

    structural = _parse_score("STRUCTURAL")
    texture = _parse_score("TEXTURE")
    context = _parse_score("CONTEXT")
    fb_match = re.search(r"FEEDBACK\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE | re.DOTALL)
    feedback = fb_match.group(1).strip() if fb_match else ""

    return {
        "score": (structural + texture + context) / 3.0,
        "structural": structural,
        "texture": texture,
        "context": context,
        "feedback": feedback,
        "raw": output_text,
    }


# Đưa ảnh hoàn chỉnh và ngữ cảnh vào Qwen để V2 kiểm tra chất lượng.
def run_target_aware_critique(
    vlm,
    critic_completed_rgb: np.ndarray,
    context_highlight_rgb: np.ndarray,
    object_name: str,
) -> dict:
    instruction = (
        f"You are a strict quality critic for mask-guided amodal appearance completion of a '{object_name}'.\n"
        "- Image 1 shows the reconstructed complete target object isolated on a white background.\n"
        "- Image 2 shows the scene where the target's visible region is inside BRIGHT GREEN and the removed "
        "foreground occluder is covered by a diagonal striped censor mask inside YELLOW.\n\n"
        "Evaluate ONLY the reconstructed target object in Image 1 on a 0-10 integer scale:\n"
        "1. STRUCTURAL: internal anatomical/structural continuity across the reconstructed region.\n"
        "2. TEXTURE: seamless color, material, and lighting transition between visible and restored parts (penalize seams, gray smudges, or hallucinated items).\n"
        "3. CONTEXT: semantic fidelity to the BRIGHT GREEN target object in Image 2 (note: if the target is a container/tray/dish, a clean EMPTY interior is 10/10 correct).\n\n"
        "Respond EXACTLY in this 4-line format:\n"
        "STRUCTURAL: <int>\n"
        "TEXTURE: <int>\n"
        "CONTEXT: <int>\n"
        "FEEDBACK: <one short actionable fix for texture/color continuity, or 'good' if seamless>"
    )

    output_text = _generate_qwen_vl(vlm, critic_completed_rgb, context_highlight_rgb, instruction, max_new_tokens=120)
    return _parse_critic_response(output_text)
