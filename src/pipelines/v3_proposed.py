"""Proposed SD1.5 experiment: structured prompts, visual guidance and strict critique."""
from PIL import Image
from src.models.sd15_ip_adapter import AdvancedInpaintPass
from src.pipelines.v2_pipeline import V2Pipeline, critic_panels
from src.vlm.micro_attributes import run_micro_reasoning
from src.vlm.strict_critic import run_strict_critic


class V3Proposed(V2Pipeline):
    def _inpaint_pipe(self, isolated, seed):
        return AdvancedInpaintPass(self.models.pipe, self.config, Image.fromarray(isolated), self.device, seed)

    def _describe(self, context, isolated, query):
        if self.config["use_micro_attributes"]:
            return run_micro_reasoning(self.models.vlm, context, isolated, query)
        return super()._describe(context, isolated, query)

    def _critique(self, panels, context, name):
        if self.config["use_strict_critic"]:
            return run_strict_critic(self.models.vlm, panels, name, self.config["critique_threshold"])
        return super()._critique(panels, context, name)
