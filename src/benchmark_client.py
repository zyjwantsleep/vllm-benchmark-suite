from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from benchmark_common import (
    build_run_id,
    ensure_results_layout,
    expand_work_items,
    load_json,
    load_prompts,
    summarize_results,
    write_csv,
)


def build_payload(prompt: str, model: str, benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": benchmark_cfg["max_tokens"],
        "temperature": benchmark_cfg["temperature"],
        "top_p": benchmark_cfg["top_p"],
        "stream": True,
    }


def tokenize_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def run_one_request(
    item: dict[str, Any],
    endpoint: str,
    headers: dict[str, str],
    model_name: str,
    benchmark_cfg: dict[str, Any],
    timeout_seconds: int,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = build_payload(item["prompt"], model_name, benchmark_cfg)
    status_code = 0
    error = ""
    generated_parts: list[str] = []
    ttft_ms = 0.0
    success = False

    try:
        with requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
            stream=True,
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("text", "")
                if text and ttft_ms == 0.0:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                if text:
                    generated_parts.append(text)
            success = True
    except Exception as exc:
        error = str(exc)

    ended = time.perf_counter()
    latency_ms = (ended - started) * 1000.0
    output_text = "".join(generated_parts)
    prompt_tokens = tokenize_len(tokenizer, item["prompt"])
    completion_tokens = tokenize_len(tokenizer, output_text) if output_text else 0
    total_tokens = prompt_tokens + completion_tokens

    return {
        "request_id": item["request_id"],
        "prompt_id": item["prompt_id"],
        "success": success,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "prompt_chars": len(item["prompt"]),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_sec": round(completion_tokens / max(latency_ms / 1000.0, 1e-9), 3),
        "total_tokens_per_sec": round(total_tokens / max(latency_ms / 1000.0, 1e-9), 3),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI-compatible serving.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--prompt-token-target", type=int, default=0)
    parser.add_argument("--gpu-metrics-path", type=str, default="")
    args = parser.parse_args()

    config = load_json(args.config)
    benchmark_cfg = dict(config["benchmark"])
    server_cfg = config["server"]
    model_cfg = config["model"]
    root_dir = args.config.parent.parent

    if args.concurrency is not None:
        benchmark_cfg["concurrency"] = args.concurrency
    if args.max_tokens is not None:
        benchmark_cfg["max_tokens"] = args.max_tokens

    prompt_path = args.dataset_path if args.dataset_path else (root_dir / benchmark_cfg["dataset_path"])
    layout = ensure_results_layout(root_dir / benchmark_cfg["results_dir"])
    prompts = load_prompts(prompt_path)
    items = expand_work_items(
        prompts,
        num_requests=int(benchmark_cfg["num_requests"]),
        repeats=int(benchmark_cfg["repeats"]),
    )

    endpoint = f"http://{server_cfg['host']}:{server_cfg['port']}/v1/completions"
    headers = {
        "Authorization": f"Bearer {server_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    model_name = model_cfg["served_model_name"]
    timeout_seconds = int(benchmark_cfg["timeout_seconds"])
    run_id = build_run_id()
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
    )

    run_meta = {
        "run_id": run_id,
        "mode": "vllm_http_stream",
        "project_name": config["project_name"],
        "endpoint": endpoint,
        "model_path": model_cfg["path"],
        "served_model_name": model_name,
        "dataset_path": str(prompt_path),
        "prompt_token_target": int(args.prompt_token_target),
        "concurrency": int(benchmark_cfg["concurrency"]),
        "num_requests": len(items),
        "max_tokens": int(benchmark_cfg["max_tokens"]),
        "temperature": benchmark_cfg["temperature"],
        "gpu_metrics_path": args.gpu_metrics_path,
    }

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(benchmark_cfg["concurrency"])) as executor:
        futures = [
            executor.submit(
                run_one_request,
                item,
                endpoint,
                headers,
                model_name,
                benchmark_cfg,
                timeout_seconds,
                tokenizer,
            )
            for item in items
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    ended_at = time.perf_counter()

    rows.sort(key=lambda row: row["request_id"])
    summary = summarize_results(rows, run_meta)
    summary["wall_clock_total_sec"] = round(ended_at - started_at, 3)
    summary["effective_requests_per_sec"] = round(len(rows) / max(ended_at - started_at, 1e-9), 3)

    raw_payload = {"summary": summary, "requests": rows}
    summary_path = layout["raw"] / f"summary_{run_id}.json"
    raw_path = layout["raw"] / f"raw_{run_id}.json"
    table_path = layout["tables"] / f"requests_{run_id}.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with raw_path.open("w", encoding="utf-8") as file:
        json.dump(raw_payload, file, ensure_ascii=False, indent=2)
    write_csv(table_path, rows)

    print(f"[benchmark] summary: {summary_path}")
    print(f"[benchmark] raw: {raw_path}")
    print(f"[benchmark] table: {table_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()