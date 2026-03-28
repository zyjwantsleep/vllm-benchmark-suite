from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


METRIC_SPECS = [
    ("effective_requests_per_sec", "Req/s", False),
    ("p50_ttft_ms", "P50 TTFT (ms)", True),
    ("p95_latency_ms", "P95 Latency (ms)", True),
]


COLORS = {
    "base": "#2563eb",
    "256": "#0f766e",
    "1024": "#dc2626",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prompt_label(meta: dict[str, Any]) -> str:
    value = int(meta.get("prompt_token_target", 0) or 0)
    return "base" if value == 0 else str(value)


def format_metric(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def latest_successful_by_combo(results_dir: Path) -> list[dict[str, Any]]:
    raw_dir = results_dir / "raw"
    latest_by_combo: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("summary_*.json")):
        summary = load_json(path)
        if int(summary.get("success_count", 0)) <= 0:
            continue
        meta = summary.get("run_meta", {})
        combo = (
            int(meta.get("prompt_token_target", 0) or 0),
            int(meta.get("concurrency", 0) or 0),
            int(meta.get("max_tokens", 0) or 0),
        )
        latest_by_combo[combo] = summary
    return [latest_by_combo[key] for key in sorted(latest_by_combo.keys())]


def write_matrix_summary_csv(results_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "matrix_summary_latest.csv"
    fieldnames = [
        "prompt",
        "concurrency",
        "max_tokens",
        "success_rate",
        "effective_requests_per_sec",
        "avg_latency_ms",
        "p50_ttft_ms",
        "p95_latency_ms",
        "avg_output_tokens_per_sec",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "run_id",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            meta = summary.get("run_meta", {})
            writer.writerow(
                {
                    "prompt": prompt_label(meta),
                    "concurrency": meta.get("concurrency", ""),
                    "max_tokens": meta.get("max_tokens", ""),
                    "success_rate": summary.get("success_rate", 0.0),
                    "effective_requests_per_sec": summary.get("effective_requests_per_sec", 0.0),
                    "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
                    "p50_ttft_ms": summary.get("p50_ttft_ms", 0.0),
                    "p95_latency_ms": summary.get("p95_latency_ms", 0.0),
                    "avg_output_tokens_per_sec": summary.get("avg_output_tokens_per_sec", 0.0),
                    "avg_prompt_tokens": summary.get("avg_prompt_tokens", 0.0),
                    "avg_completion_tokens": summary.get("avg_completion_tokens", 0.0),
                    "run_id": meta.get("run_id", ""),
                }
            )
    return output_path


def render_heatmaps(results_dir: Path, summaries: list[dict[str, Any]]) -> Path | None:
    if not summaries:
        return None

    chart_dir = results_dir / "charts"
    prompt_groups = sorted({prompt_label(summary.get("run_meta", {})) for summary in summaries}, key=lambda x: (x != "base", int(x) if x.isdigit() else -1))
    conc_values = sorted({int(summary["run_meta"]["concurrency"]) for summary in summaries})
    max_token_values = sorted({int(summary["run_meta"]["max_tokens"]) for summary in summaries})
    index = {
        (prompt_label(summary.get("run_meta", {})), int(summary["run_meta"]["concurrency"]), int(summary["run_meta"]["max_tokens"])): summary
        for summary in summaries
    }

    fig, axes = plt.subplots(len(prompt_groups), len(METRIC_SPECS), figsize=(5.4 * len(METRIC_SPECS), 3.8 * len(prompt_groups)), squeeze=False)

    for row_idx, prompt in enumerate(prompt_groups):
        for col_idx, (metric_key, metric_title, lower_is_better) in enumerate(METRIC_SPECS):
            axis = axes[row_idx][col_idx]
            matrix: list[list[float]] = []
            for conc in conc_values:
                row: list[float] = []
                for max_tokens in max_token_values:
                    summary = index.get((prompt, conc, max_tokens))
                    row.append(float(summary.get(metric_key, 0.0)) if summary else float("nan"))
                matrix.append(row)

            cmap = "RdYlGn_r" if lower_is_better else "YlGnBu"
            image = axis.imshow(matrix, aspect="auto", cmap=cmap)
            axis.set_title(f"prompt={prompt} | {metric_title}")
            axis.set_xticks(range(len(max_token_values)), [str(v) for v in max_token_values])
            axis.set_yticks(range(len(conc_values)), [str(v) for v in conc_values])
            axis.set_xlabel("Max Tokens")
            axis.set_ylabel("Concurrency")

            for y, conc in enumerate(conc_values):
                for x, max_tokens in enumerate(max_token_values):
                    summary = index.get((prompt, conc, max_tokens))
                    text = "-" if summary is None else format_metric(float(summary.get(metric_key, 0.0)))
                    axis.text(x, y, text, ha="center", va="center", fontsize=8, color="black")

            cbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    fig.suptitle("Matrix Heatmap Overview", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = chart_dir / "benchmark_compare.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pareto(results_dir: Path, summaries: list[dict[str, Any]]) -> Path | None:
    if len(summaries) < 2:
        return None
    chart_dir = results_dir / "charts"
    fig, ax = plt.subplots(figsize=(10.5, 7))

    for summary in summaries:
        meta = summary.get("run_meta", {})
        prompt = prompt_label(meta)
        x = float(summary.get("effective_requests_per_sec", 0.0))
        y = float(summary.get("p95_latency_ms", 0.0))
        size = max(float(summary.get("p50_ttft_ms", 0.0)), 1.0) * 0.35
        label = f"{prompt}-c{meta.get('concurrency')}-t{meta.get('max_tokens')}"
        ax.scatter(x, y, s=size, alpha=0.72, color=COLORS.get(prompt, "#7c3aed"), edgecolors="black", linewidths=0.5)
        ax.text(x, y, label, fontsize=8, ha="left", va="bottom")

    ax.set_title("Pareto View: Throughput vs Tail Latency")
    ax.set_xlabel("Effective Requests/sec")
    ax.set_ylabel("P95 Latency (ms)")
    ax.grid(alpha=0.25)

    output_path = chart_dir / "benchmark_pareto.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_prompt_trends(results_dir: Path, summaries: list[dict[str, Any]]) -> Path | None:
    if len(summaries) < 2:
        return None
    chart_dir = results_dir / "charts"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    prompts = sorted({prompt_label(s.get("run_meta", {})) for s in summaries}, key=lambda x: (x != "base", int(x) if x.isdigit() else -1))
    for prompt in prompts:
        subset = [s for s in summaries if prompt_label(s.get("run_meta", {})) == prompt]
        subset.sort(key=lambda s: (int(s["run_meta"]["concurrency"]), int(s["run_meta"]["max_tokens"])))
        xs = [f"c{s['run_meta']['concurrency']}-t{s['run_meta']['max_tokens']}" for s in subset]
        axes[0].plot(xs, [float(s.get("effective_requests_per_sec", 0.0)) for s in subset], marker="o", linewidth=1.8, label=prompt, color=COLORS.get(prompt, "#7c3aed"))
        axes[1].plot(xs, [float(s.get("p50_ttft_ms", 0.0)) for s in subset], marker="o", linewidth=1.8, label=prompt, color=COLORS.get(prompt, "#7c3aed"))
        axes[2].plot(xs, [float(s.get("p95_latency_ms", 0.0)) for s in subset], marker="o", linewidth=1.8, label=prompt, color=COLORS.get(prompt, "#7c3aed"))

    titles = ["Req/s Trends", "P50 TTFT Trends", "P95 Latency Trends"]
    ylabels = ["Req/s", "TTFT (ms)", "Latency (ms)"]
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=65)
        axis.grid(alpha=0.25)
        axis.legend(title="Prompt")

    fig.tight_layout()
    output_path = chart_dir / "benchmark_prompt_trends.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def top_rows_html(summaries: list[dict[str, Any]], title: str, key: str, reverse: bool) -> str:
    ranked = sorted(summaries, key=lambda item: float(item.get(key, 0.0)), reverse=reverse)[:8]
    rows: list[str] = []
    for summary in ranked:
        meta = summary.get("run_meta", {})
        rows.append(
            "<tr>"
            f"<td>{prompt_label(meta)}</td>"
            f"<td>{meta.get('concurrency')}</td>"
            f"<td>{meta.get('max_tokens')}</td>"
            f"<td>{summary.get('effective_requests_per_sec', 0):.2f}</td>"
            f"<td>{summary.get('p50_ttft_ms', 0):.1f}</td>"
            f"<td>{summary.get('p95_latency_ms', 0):.1f}</td>"
            "</tr>"
        )
    return (
        f"<div class='card'><h2>{title}</h2><table><thead><tr><th>Prompt</th><th>Concurrency</th><th>Max Tokens</th>"
        "<th>Req/s</th><th>P50 TTFT</th><th>P95 Latency</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def write_html_report(
    results_dir: Path,
    compare_plot: Path | None,
    pareto_plot: Path | None,
    trend_plot: Path | None,
    summary_csv: Path | None,
    summaries: list[dict[str, Any]],
) -> Path:
    report_path = results_dir / "reports" / "benchmark_report_latest.html"
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>vLLM Matrix Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }}
    .card {{ background: rgba(255,255,255,0.92); border: 1px solid #dbe4ff; border-radius: 14px; padding: 18px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; }}
    .pill {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 10px 12px; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
    img {{ width: 100%; border: 1px solid #dbe4ff; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>vLLM Matrix Report</h1>
  <div class=\"card\">
    <div class=\"meta\">
      <div class=\"pill\"><strong>Results Dir</strong><br><code>{results_dir}</code></div>
      <div class=\"pill\"><strong>Successful Combos</strong><br>{len(summaries)}</div>
      <div class=\"pill\"><strong>Prompt Tiers</strong><br>{', '.join(sorted({prompt_label(s.get('run_meta', {})) for s in summaries})) or 'n/a'}</div>
      <div class=\"pill\"><strong>Summary CSV</strong><br><code>{summary_csv.name if summary_csv else 'n/a'}</code></div>
    </div>
  </div>
  <div class=\"card\">
    <h2>Matrix Heatmaps</h2>
    <p>按 prompt 长度分层，直接比较 Req/s、P50 TTFT、P95 Latency。</p>
    {"<img src='../charts/benchmark_compare.png' alt='matrix heatmaps'>" if compare_plot else "<p>Need at least 2 successful runs to compare.</p>"}
  </div>
  <div class=\"card\">
    <h2>Prompt Trends</h2>
    <p>把 base / 256 / 1024 三类 prompt 在不同 concurrency 和 max_tokens 下的趋势放到一起。</p>
    {"<img src='../charts/benchmark_prompt_trends.png' alt='prompt trends'>" if trend_plot else "<p>Need at least 2 successful runs to render trend view.</p>"}
  </div>
  <div class=\"card\">
    <h2>Pareto View</h2>
    <p>看吞吐与尾延迟的权衡，点越靠右下通常越值得优先看。</p>
    {"<img src='../charts/benchmark_pareto.png' alt='pareto'>" if pareto_plot else "<p>Need at least 2 successful runs to render pareto view.</p>"}
  </div>
  {top_rows_html(summaries, 'Top Throughput Configs', 'effective_requests_per_sec', True)}
  {top_rows_html(summaries, 'Lowest TTFT Configs', 'p50_ttft_ms', False)}
  {top_rows_html(summaries, 'Lowest P95 Latency Configs', 'p95_latency_ms', False)}
</body>
</html>
"""
    with report_path.open("w", encoding="utf-8") as file:
        file.write(html)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark charts and HTML report.")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = latest_successful_by_combo(args.results_dir)
    summary_csv = write_matrix_summary_csv(args.results_dir, summaries) if summaries else None
    compare_plot = render_heatmaps(args.results_dir, summaries)
    pareto_plot = render_pareto(args.results_dir, summaries)
    trend_plot = render_prompt_trends(args.results_dir, summaries)
    report_path = write_html_report(args.results_dir, compare_plot, pareto_plot, trend_plot, summary_csv, summaries)

    if compare_plot:
        print(f"[visualize] heatmap chart: {compare_plot}")
    if pareto_plot:
        print(f"[visualize] pareto chart: {pareto_plot}")
    if trend_plot:
        print(f"[visualize] trend chart: {trend_plot}")
    if summary_csv:
        print(f"[visualize] matrix summary: {summary_csv}")
    print(f"[visualize] report: {report_path}")


if __name__ == "__main__":
    main()