"""SQLite audit trail: every routed request gets a row."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "requests.db"

_SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prompt_hash TEXT,
    complexity_tier INTEGER,
    routed_model TEXT,
    cost_usd REAL,
    latency_ms REAL,
    quality_score REAL,
    escalated INTEGER
)"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_request(
    prompt: str,
    tier: int,
    model_name: str,
    cost_usd: float,
    latency_ms: float,
    quality_score: float | None,
    escalated: bool,
) -> None:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    conn = _conn()
    conn.execute(
        "INSERT INTO requests (timestamp, prompt_hash, complexity_tier, routed_model, "
        "cost_usd, latency_ms, quality_score, escalated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            prompt_hash,
            tier,
            model_name,
            cost_usd,
            latency_ms,
            quality_score,
            int(escalated),
        ),
    )
    conn.commit()
    conn.close()


def cost_summary() -> dict:
    """Total cost vs. hypothetical all-gpt-4o cost, routing distribution, escalation rate."""
    from src.registry import get_model

    conn = _conn()
    rows = conn.execute(
        "SELECT complexity_tier, routed_model, cost_usd, escalated FROM requests"
    ).fetchall()
    conn.close()

    if not rows:
        return {"total_requests": 0, "total_cost_usd": 0.0, "hypothetical_gpt4o_cost_usd": 0.0,
                 "savings_pct": 0.0, "routing_distribution": {}, "escalation_rate": 0.0}

    total_cost = sum(r[2] for r in rows)
    escalation_rate = sum(r[3] for r in rows) / len(rows)

    distribution: dict[str, int] = {}
    for _, model, _, _ in rows:
        distribution[model] = distribution.get(model, 0) + 1

    # Hypothetical: what if every request had gone to gpt-4o instead?
    gpt4o = get_model("gpt-4o")
    avg_input_tokens, avg_output_tokens = 500, 300  # rough baseline for estimation
    hypothetical_cost = len(rows) * gpt4o.cost_for(avg_input_tokens, avg_output_tokens)

    savings_pct = 0.0
    if hypothetical_cost > 0:
        savings_pct = max(0.0, (hypothetical_cost - total_cost) / hypothetical_cost * 100)

    return {
        "total_requests": len(rows),
        "total_cost_usd": round(total_cost, 4),
        "hypothetical_gpt4o_cost_usd": round(hypothetical_cost, 4),
        "savings_pct": round(savings_pct, 1),
        "routing_distribution": distribution,
        "escalation_rate": round(escalation_rate, 3),
    }
