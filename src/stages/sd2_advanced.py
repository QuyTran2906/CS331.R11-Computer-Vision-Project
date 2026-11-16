"""Compatibility import; implementation lives in src.models.sd15_ip_adapter."""
import sys
from src.models import sd15_ip_adapter as _implementation
sys.modules[__name__] = _implementation
