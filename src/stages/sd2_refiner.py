"""Historical import names for ROI inpainting."""
from src.stages.inpaint_refiner import (
    run_inpaint_roi_pass as _run_sd2_roi_pass,
    refine_roi_blind as refine_roi_sd2_blind,
    refine_exact_mask as refine_exact_mask_sd2_oracle,
)
