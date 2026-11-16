"""Compatibility import; direct inpainting now lives in v1_direct."""
import sys
from src.pipelines import v1_direct as _implementation
sys.modules[__name__] = _implementation
