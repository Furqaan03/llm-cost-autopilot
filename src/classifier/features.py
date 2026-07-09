"""Feature extraction for the complexity classifier — no LLM calls, pure heuristics."""
from __future__ import annotations

import re

import numpy as np

_ANALYSIS_WORDS = re.compile(r"\b(analyze|compare|evaluate|synthesize|critique|justify|reason)\b", re.I)
_CONSTRAINT_WORDS = re.compile(r"\b(must|should|ensure|require|only|exactly|within|between)\b", re.I)
_FORMAT_WORDS = re.compile(r"\b(json|table|bullet|markdown|xml|csv|schema)\b", re.I)
_CONTEXT_MARKERS = re.compile(r"(```|<context>|based on the following|given the text)", re.I)

FEATURE_NAMES = [
    "token_count",
    "has_analysis_instruction",
    "constraint_count",
    "has_context_provided",
    "format_complexity",
]


def extract_features(prompt: str) -> np.ndarray:
    token_count = len(prompt.split())
    has_analysis = 1 if _ANALYSIS_WORDS.search(prompt) else 0
    constraint_count = len(_CONSTRAINT_WORDS.findall(prompt))
    has_context = 1 if _CONTEXT_MARKERS.search(prompt) else 0
    format_complexity = len(_FORMAT_WORDS.findall(prompt))

    return np.array([[token_count, has_analysis, constraint_count, has_context, format_complexity]], dtype=float)
