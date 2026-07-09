# LLM Cost Autopilot

An intelligent routing layer that sits in front of OpenAI, Anthropic, and a
local Ollama model, classifies every incoming request's complexity, routes it
to the cheapest model that can handle it, and asynchronously verifies routing
decisions against a top-tier reference model — auto-escalating when the cheap
model's answer diverges too far.

## Why this exists

Every team running LLMs at scale sends far more traffic to expensive models
than the task requires. This treats model selection as a cost-optimization
problem instead of a hardcoded choice — the same way a CDN doesn't serve
every request from the origin server.

## Architecture

```
src/registry.py            model registry: real pricing, latency, quality tier per model
src/provider_client.py     unified send_request() across OpenAI / Anthropic / Ollama
src/classifier/
  build_dataset.py         generates the 240-example labeled complexity dataset (3 tiers)
  features.py               heuristic feature extraction (token count, analysis-instruction
                             signal, constraint count, context markers, format complexity)
  train.py                  trains + evaluates a logistic regression tier classifier
  predict.py                runtime tier prediction + routing_config.yaml lookup
  routing_config.yaml       tier -> model name, hot-editable without redeploy
src/verifier/quality.py    async agreement-scoring against a reference model + escalation
src/logging_store.py       SQLite audit trail + cost-savings aggregation
src/api/main.py            FastAPI: POST /v1/completions, GET /v1/models, GET /v1/stats,
                            PUT /v1/routing-config
```

## Design decisions

- **The classifier dataset is templated, not hand-typed one-by-one.** 24 templates
  (8 per tier) x 8 fillers = 240 examples, each template a genuine instance of that
  tier's defining trait per the guide's tier definitions (reformatting/extraction vs.
  summarization/classification vs. multi-step reasoning/synthesis). Reviewable in
  `data/labeled_prompts.jsonl` before training — not a black box.
- **91.7% held-out accuracy** on a stratified 80/20 split, comfortably above the
  guide's 80% bar for a V1 routing skeleton. Confusion matrix shows tier 1/2 as the
  only meaningful confusion pair (adjacent tiers, expected); tier 3 is never confused.
- **Agreement scoring uses lexical overlap, not a third LLM call.** Verifying a cheap
  model's output by paying for *another* LLM call to grade the comparison defeats the
  cost-saving purpose. A token-overlap heuristic against the reference model's own
  output is free and good enough to catch large divergences.
- **The caller never picks a model.** `POST /v1/completions` takes a prompt, returns
  a response with metadata showing which model was actually used and why — mirroring
  how a real routing layer would be consumed by application code.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env       # fill in OPENAI_API_KEY (ANTHROPIC_API_KEY optional)
python -m src.classifier.train
```

## Running

```bash
uvicorn src.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "verify": true}'

curl http://localhost:8000/v1/stats
curl http://localhost:8000/v1/models
```

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Status

Phases 1-5 complete: unified provider client, trained complexity classifier
with routing map, async verification/escalation, SQLite cost logging, FastAPI
service. Phase 6 (load test + case study write-up) is a manual step once
enough real traffic has been routed through `/v1/completions`.
