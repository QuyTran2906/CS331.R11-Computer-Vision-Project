"""Compatibility import; implementation lives in src.models.memory_manager."""
import sys
from src.models import memory_manager as _implementation
sys.modules[__name__] = _implementation
