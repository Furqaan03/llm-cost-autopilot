"""FastAPI service: the router decides the model, not the caller."""
from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.classifier.predict import ROUTING_CONFIG_PATH, route_to_model_name
from src.logging_store import cost_summary, log_request
from src.provider_client import send_request
from src.registry import REGISTRY, get_model
from src.verifier.quality import verify_response

load_dotenv()

app = FastAPI(title="LLM Cost Autopilot")


class CompletionRequest(BaseModel):
    prompt: str
    verify: bool = False


class CompletionResponse(BaseModel):
    output_text: str
    routed_model: str
    complexity_tier: int
    cost_usd: float
    latency_ms: float
    escalated: bool
    escalation_reason: str | None = None


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(req: CompletionRequest) -> CompletionResponse:
    tier, model_name = route_to_model_name(req.prompt)
    config = get_model(model_name)
    response = send_request(req.prompt, config)

    escalated = False
    escalation_reason = None
    final_response = response
    final_model_name = model_name

    if req.verify:
        verification = verify_response(req.prompt, response)
        if verification.escalation_recommended:
            escalated = True
            escalation_reason = f"agreement score {verification.agreement_score:.2f} below threshold"
            final_response = verification.reference_response
            final_model_name = verification.reference_model

    log_request(
        prompt=req.prompt,
        tier=tier,
        model_name=final_model_name,
        cost_usd=final_response.cost_usd,
        latency_ms=final_response.latency_ms,
        quality_score=None,
        escalated=escalated,
    )

    return CompletionResponse(
        output_text=final_response.output_text,
        routed_model=final_model_name,
        complexity_tier=tier,
        cost_usd=final_response.cost_usd,
        latency_ms=final_response.latency_ms,
        escalated=escalated,
        escalation_reason=escalation_reason,
    )


@app.get("/v1/models")
def list_models() -> dict:
    return {
        name: {
            "provider": cfg.provider.value,
            "cost_per_1m_input": cfg.cost_per_1m_input,
            "cost_per_1m_output": cfg.cost_per_1m_output,
            "quality_tier": cfg.quality_tier.value,
        }
        for name, cfg in REGISTRY.items()
    }


@app.get("/v1/stats")
def stats() -> dict:
    return cost_summary()


class RoutingUpdate(BaseModel):
    tier: int
    model_name: str


@app.put("/v1/routing-config")
def update_routing(update: RoutingUpdate) -> dict:
    if update.model_name not in REGISTRY:
        return {"error": f"Unknown model '{update.model_name}'"}
    config = yaml.safe_load(ROUTING_CONFIG_PATH.read_text(encoding="utf-8"))
    config[update.tier] = update.model_name
    ROUTING_CONFIG_PATH.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"status": "updated", "tier": update.tier, "model_name": update.model_name}
