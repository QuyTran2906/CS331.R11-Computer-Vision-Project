# Phạm vi: Chỉ V3: thay các điểm mở rộng của V2; ablation tắt đường ảnh tham chiếu nhưng vẫn giữ
# latent blending và critic.
"""Proposed SD1.5 experiment: structured prompts, visual guidance and strict critique."""
from PIL import Image
from src.models.sd15_ip_adapter import AdvancedInpaintPass
from src.pipelines.v2_pipeline import V2Pipeline, critic_panels
from src.vlm.micro_attributes import run_micro_reasoning
from src.vlm.strict_critic import run_strict_critic


# Giữ vòng lặp V2, thay ba điểm mở rộng để bật các cơ chế V3.
class V3Proposed(V2Pipeline):
    # Gắn crop visible làm tham chiếu và cấp callback trộn latent cho lần sinh này.
    def _inpaint_pipe(self, isolated, seed):
        return AdvancedInpaintPass(self.models.pipe, self.config, Image.fromarray(isolated), self.device, seed)

    # Dùng mô tả thuộc tính có cấu trúc thay prompt cơ bản khi bật cờ.
    def _describe(self, context, isolated, query):
        if self.config["use_micro_attributes"]:
            return run_micro_reasoning(self.models.vlm, context, isolated, query)
        return super()._describe(context, isolated, query)

    # Dùng hai tiêu chí riêng và quyết định PASS/REJECT có cấu trúc.
    def _critique(self, panels, context, name):
        if self.config["use_strict_critic"]:
            return run_strict_critic(self.models.vlm, panels, name, self.config["critique_threshold"])
        return super()._critique(panels, context, name)
