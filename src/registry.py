"""Model registry: real per-token pricing and quality tiers for every routable model.

Prices are USD per 1M tokens, as published by each provider (checked 2026-07).
Update these when providers change pricing — the routing/cost math reads directly
from this table, nothing is hardcoded downstream.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    provider: Provider
    model_id: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    avg_latency_ms: float
    quality_tier: QualityTier

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.cost_per_1m_input + (
            output_tokens / 1_000_000
        ) * self.cost_per_1m_output


REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI, model_id="gpt-4o",
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
        avg_latency_ms=1800, quality_tier=QualityTier.HIGH,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI, model_id="gpt-4o-mini",
        cost_per_1m_input=0.15, cost_per_1m_output=0.60,
        avg_latency_ms=900, quality_tier=QualityTier.MEDIUM,
    ),
    "claude-sonnet": ModelConfig(
        provider=Provider.ANTHROPIC, model_id="claude-sonnet-4-5",
        cost_per_1m_input=3.00, cost_per_1m_output=15.00,
        avg_latency_ms=1600, quality_tier=QualityTier.HIGH,
    ),
    "claude-haiku": ModelConfig(
        provider=Provider.ANTHROPIC, model_id="claude-haiku-4-5",
        cost_per_1m_input=0.80, cost_per_1m_output=4.00,
        avg_latency_ms=700, quality_tier=QualityTier.MEDIUM,
    ),
    "llama3-local": ModelConfig(
        provider=Provider.OLLAMA, model_id="llama3",
        cost_per_1m_input=0.0, cost_per_1m_output=0.0,
        avg_latency_ms=1200, quality_tier=QualityTier.LOW,
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name]
