"""Compatibility import; implementation lives in src.eval.metrics."""
import sys
from src.eval import metrics as _implementation
sys.modules[__name__] = _implementation
