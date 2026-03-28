from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def expand_work_items(prompts: list[dict[str, str]], num_requests: int, repeats: int) -> list[dict[str, Any]]:
    total = max(num_requests, len(prompts)) * max(repeats, 1)
    items: list[dict[str, Any]] = []
    for index in range(total):
        prompt_item = prompts[index % len(prompts)]
        items.append(
            {
                "request_id": f"req_{index + 1:04d}",
                "prompt_id": prompt_item.get("id", f"prompt_{index + 1:04d}"),
                "prompt": prompt_item["prompt"],
            }
        )
    if num_requests > 0:
        return items[: num_requests * max(repeats, 1)]
    return items


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_results_layout(results_root: Path) -> dict[str, Path]:
    layout = {
        "root": results_root,
        "logs": results_root / "logs",
        "raw": results_root / "raw",
        "tables": results_root / "tables",
        "charts": results_root / "charts",
        "reports": results_root / "reports",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def build_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def summarize_results(rows: list[dict[str, Any]], run_meta: dict[str, Any]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["success"]]
    latencies = [row["latency_ms"] for row in success_rows]
    ttft_values = [row["ttft_ms"] for row in success_rows if row["ttft_ms"] > 0]
    output_tps = [row["output_tokens_per_sec"] for row in success_rows]
    total_tps = [row["total_tokens_per_sec"] for row in success_rows]
    prompt_tokens = [row["prompt_tokens"] for row in success_rows]
    completion_tokens = [row["completion_tokens"] for row in success_rows]
    statuses = Counter(str(row["status_code"]) for row in rows)

    return {
        "run_meta": run_meta,
        "request_count": len(rows),
        "success_count": len(success_rows),
        "success_rate": round(len(success_rows) / len(rows), 4) if rows else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.50), 3) if latencies else 0.0,
        "p90_latency_ms": round(percentile(latencies, 0.90), 3) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 3) if latencies else 0.0,
        "avg_ttft_ms": round(statistics.mean(ttft_values), 3) if ttft_values else 0.0,
        "p50_ttft_ms": round(percentile(ttft_values, 0.50), 3) if ttft_values else 0.0,
        "p90_ttft_ms": round(percentile(ttft_values, 0.90), 3) if ttft_values else 0.0,
        "avg_output_tokens_per_sec": round(statistics.mean(output_tps), 3) if output_tps else 0.0,
        "avg_total_tokens_per_sec": round(statistics.mean(total_tps), 3) if total_tps else 0.0,
        "avg_prompt_tokens": round(statistics.mean(prompt_tokens), 3) if prompt_tokens else 0.0,
        "avg_completion_tokens": round(statistics.mean(completion_tokens), 3) if completion_tokens else 0.0,
        "status_breakdown": dict(statuses),
    }