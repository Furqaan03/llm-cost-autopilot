"""Unified interface across OpenAI, Anthropic, and Ollama behind one call shape."""
from __future__ import annotations

import time

import httpx
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from src.registry import ModelConfig, Provider

_openai_client: OpenAI | None = None
_anthropic_client: Anthropic | None = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _anthropic() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


class Response(BaseModel):
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str
    provider: str


def _send_openai(prompt: str, config: ModelConfig) -> tuple[str, int, int]:
    resp = _openai().chat.completions.create(
        model=config.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _send_anthropic(prompt: str, config: ModelConfig) -> tuple[str, int, int]:
    resp = _anthropic().messages.create(
        model=config.model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _send_ollama(prompt: str, config: ModelConfig, base_url: str = "http://localhost:11434") -> tuple[str, int, int]:
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url}/api/generate",
            json={"model": config.model_id, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
    return (
        data.get("response", ""),
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


def send_request(prompt: str, config: ModelConfig) -> Response:
    """Sends `prompt` to whichever provider `config` points at and returns a
    standardized Response regardless of provider."""
    start = time.perf_counter()

    if config.provider == Provider.OPENAI:
        text, in_tok, out_tok = _send_openai(prompt, config)
    elif config.provider == Provider.ANTHROPIC:
        text, in_tok, out_tok = _send_anthropic(prompt, config)
    elif config.provider == Provider.OLLAMA:
        text, in_tok, out_tok = _send_ollama(prompt, config)
    else:
        raise ValueError(f"Unhandled provider: {config.provider}")

    latency_ms = (time.perf_counter() - start) * 1000

    return Response(
        output_text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=config.cost_for(in_tok, out_tok),
        model_id=config.model_id,
        provider=config.provider.value,
    )
