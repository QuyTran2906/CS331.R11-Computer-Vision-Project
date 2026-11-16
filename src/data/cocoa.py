"""Compatibility import; implementation lives in src.data.cocoa_loader."""
import sys
from src.data import cocoa_loader as _implementation
sys.modules[__name__] = _implementation
