"""Async quality verification: compares a cheap model's output against the top-tier model."""
from __future__ import annotations

import json

from pydantic import BaseModel

from src.provider_client import Response, send_request
from src.registry import get_model


class VerificationResult(BaseModel):
    agreement_score: float  # 0-1, how well the cheap output matches the reference
    escalation_recommended: bool
    reference_model: str
    reference_response: Response


AGREEMENT_THRESHOLD = 0.7


def _score_agreement(cheap_output: str, reference_output: str) -> float:
    """Lightweight lexical-overlap agreement score. Cheap, no extra LLM call needed
    to score the comparison itself — avoids a third API call per verification."""
    cheap_tokens = set(cheap_output.lower().split())
    ref_tokens = set(reference_output.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(cheap_tokens & ref_tokens)
    return overlap / len(ref_tokens)


def verify_response(prompt: str, cheap_response: Response, reference_model_name: str = "gpt-4o") -> VerificationResult:
    """Re-runs the prompt on the top-tier reference model and scores agreement.
    Meant to run async/after the user already has their (cheap) response."""
    reference_config = get_model(reference_model_name)
    reference_response = send_request(prompt, reference_config)

    score = _score_agreement(cheap_response.output_text, reference_response.output_text)

    return VerificationResult(
        agreement_score=score,
        escalation_recommended=score < AGREEMENT_THRESHOLD,
        reference_model=reference_model_name,
        reference_response=reference_response,
    )
