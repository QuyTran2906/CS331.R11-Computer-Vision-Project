"""Archived SD2 Direct configuration and artifact names."""

from src.pipelines.v1_direct import DirectInpaintCompleter, DEFAULT_PROMPT, prepare_direct_inputs


class SD2DirectCompleter(DirectInpaintCompleter):
    baseline_name = "SD2 Direct"
    settings_filename = "sd2_settings.json"
    timing_key = "sd2_seconds"
