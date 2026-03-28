from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization_gpu",
    "utilization_memory",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_gpu",
    "power_draw_w",
]


def query_once() -> list[list[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample GPU metrics with nvidia-smi.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        file.flush()
        try:
            while True:
                for row in query_once():
                    writer.writerow(row)
                file.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()