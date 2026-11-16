"""Compatibility import for the shared appearance metrics."""
import sys
from src.eval import appearance_metrics as _implementation
sys.modules[__name__] = _implementation
