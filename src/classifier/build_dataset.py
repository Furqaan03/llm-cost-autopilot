"""Builds the labeled complexity-tier dataset (~240 examples across 3 tiers).

Templated rather than hand-typed one-by-one, but every template is a real
example of that tier's defining characteristics (per the guide's tier
definitions), and outputs are deterministic and reviewable in
data/labeled_prompts.jsonl before training.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "labeled_prompts.jsonl"

# Tier 1: simple — reformatting, extraction, basic Q&A from provided context.
TIER1_TEMPLATES = [
    "What is the capital of {x}?",
    "Extract the email address from this text: contact us at {x}@example.com",
    "Convert '{x}' to uppercase.",
    "What year did {x} happen?",
    "List the days of the week.",
    "Reformat this date to YYYY-MM-DD: {x}",
    "What is {x} plus {x}?",
    "Translate 'hello' to {x}.",
    "Given the text 'The sky is {x}', what color is the sky?",
    "Extract the phone number from: call me at 555-{x}-1234",
]

# Tier 2: moderate — summarization, classification, structured analysis.
TIER2_TEMPLATES = [
    "Summarize the following customer review in two sentences: {x}",
    "Classify this support ticket as billing, technical, or account: {x}",
    "Given the following sales data, identify the top-performing region: {x}",
    "Compare the pricing of Plan A and Plan B based on this table: {x}",
    "Extract and structure the key entities (names, dates, amounts) from: {x}",
    "Given this meeting transcript, list the three main action items: {x}",
    "Categorize the following list of expenses by type: {x}",
    "Summarize the sentiment of these product reviews: {x}",
    "Given this schema, generate a JSON object matching the structure: {x}",
    "Identify the main argument and one counterargument in this paragraph: {x}",
]

# Tier 3: complex — multi-step reasoning, creative generation, nuanced judgment calls.
TIER3_TEMPLATES = [
    "Analyze the following business scenario and recommend a strategy, justifying each tradeoff: {x}",
    "Given these three conflicting stakeholder requirements, synthesize a compromise proposal and explain your reasoning: {x}",
    "Critique this argument for logical fallacies and rewrite it to be more rigorous: {x}",
    "Given this incomplete dataset, infer the most likely missing values and explain your reasoning chain: {x}",
    "Write a nuanced performance review for an employee balancing praise and constructive criticism given: {x}",
    "Evaluate these two competing technical architectures across five dimensions and recommend one with justification: {x}",
    "Given this legal clause, identify all edge cases where it could be misinterpreted and propose amended language: {x}",
    "Design a multi-step experiment to test the hypothesis that {x}, including controls and expected confounds.",
    "Given this ambiguous customer complaint, reason through three possible root causes and recommend which to investigate first: {x}",
    "Synthesize these five research abstracts into a coherent literature review identifying open questions: {x}",
]

_FILLERS = [
    "France", "Q3 revenue", "the merger", "onboarding delays", "the outage",
    "user retention", "the API redesign", "customer churn", "the pricing model",
    "the security incident", "team velocity", "the migration plan", "vendor lock-in",
    "the refund policy", "engagement metrics", "the acquisition", "compliance requirements",
    "the feature rollout", "supply chain risk", "the hiring freeze", "budget overruns",
    "the data breach", "market expansion", "the partnership terms",
]


def build() -> None:
    rows = []
    for tier, templates in [(1, TIER1_TEMPLATES), (2, TIER2_TEMPLATES), (3, TIER3_TEMPLATES)]:
        for template in templates:
            for filler in _FILLERS[:8]:
                prompt = template.format(x=filler)
                rows.append({"prompt": prompt, "tier": tier})

    DATA_PATH.parent.mkdir(exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} labeled examples to {DATA_PATH}")


if __name__ == "__main__":
    build()
