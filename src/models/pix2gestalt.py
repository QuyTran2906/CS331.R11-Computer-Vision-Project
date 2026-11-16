"""Compatibility import; implementation lives in src.models.pix2gestalt_prior."""
import sys
from src.models import pix2gestalt_prior as _implementation
sys.modules[__name__] = _implementation
