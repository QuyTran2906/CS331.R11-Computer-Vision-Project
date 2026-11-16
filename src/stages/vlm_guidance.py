"""Compatibility import; implementation lives in src.vlm.qwen_basic."""
import sys
from src.vlm import qwen_basic as _implementation
sys.modules[__name__] = _implementation
