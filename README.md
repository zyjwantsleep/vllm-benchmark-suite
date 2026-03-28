# vLLM Benchmark Suite

> A lightweight local benchmark suite for vLLM serving on Linux / WSL2.

用于测试本地 vLLM 服务在不同 `prompt` 长度、并发度、输出长度下的表现，输出请求级结果、GPU 采样、热力图、趋势图和 HTML 报告。

## 目录

- [项目概览](#项目概览)
- [环境要求](#环境要求)
- [vLLM 安装说明](#vllm-安装说明)
- [快速启动](#快速启动)
- [脚本说明](#脚本说明)
- [配置说明](#配置说明)
- [输出结果](#输出结果)
- [图表怎么看](#图表怎么看)
- [常见问题](#常见问题)

## 项目概览

### 这个项目能做什么

- 启动本地 vLLM OpenAI 兼容服务
- 跑单组 benchmark
- 跑矩阵 benchmark
- 比较不同 prompt 长度、并发度、输出长度
- 记录 TTFT、延迟、吞吐、GPU 使用情况
- 生成热力图、趋势图、Pareto 图和 HTML 报告

### 适用环境

- Linux
- WSL2

### 默认模型路径

默认配置里使用的是：

```text
/home/jeje/models/Qwen2.5-1.5B-Instruct
```

这只是默认值。你可以改成任意本地模型路径，项目本身不依赖 Qwen。

### 目录结构

```text
.
├─ configs/
├─ prompts/
├─ results/
├─ scripts/
└─ src/
```

核心目录说明：

| 目录 | 作用 |
|---|---|
| `configs/` | benchmark 和服务配置 |
| `prompts/` | 短 prompt 和自动生成的长 prompt |
| `results/` | 日志、原始结果、图表、报告 |
| `scripts/` | 一键启动和执行脚本 |
| `src/` | benchmark、可视化、长 prompt 生成代码 |

## 环境要求

> 推荐先保证 vLLM 在你的机器上已经可正常运行，再使用这个 benchmark 项目。

### 基础要求

| 项目 | 建议 |
|---|---|
| OS | Linux / WSL2 |
| Python | 3.12 |
| GPU | NVIDIA GPU |
| CUDA | 可用 |
| 系统命令 | `nvidia-smi` 可执行 |

### Python 依赖

```bash
pip install vllm transformers requests matplotlib
```

### 推荐虚拟环境

```bash
python3.12 -m venv ~/vllm312
source ~/vllm312/bin/activate
pip install --upgrade pip
pip install vllm transformers requests matplotlib
```

## vLLM 安装说明

### 常规显卡

如果你的 GPU 和 CUDA 已被官方 wheel 支持，通常可以直接安装：

```bash
pip install vllm
```

### RTX 50 系列 / 新架构显卡

> 50 系列用户建议自行编译 vLLM，不要默认依赖预编译 wheel。

原因通常是：
- 新架构支持节奏较快，wheel 不一定完全匹配
- CUDA / PyTorch / 编译环境和目标架构需要对齐
- 源码编译往往更稳定

如果你已经能执行：

```bash
python -m vllm.entrypoints.openai.api_server --help
```

说明 vLLM 基本已经可用了，本项目就可以直接使用。

## 快速启动

> 最短流程：启动服务 -> 跑 benchmark -> 生成报告。

### 1. 进入项目目录

```bash
cd /path/to/vllm-benchmark-suite
```

当前这份项目在你的机器上是：

```bash
cd /home/jeje/models/qwen25_1p5b_vllm_bench
```

### 2. 激活虚拟环境

```bash
source ~/vllm312/bin/activate
```

如果你进入 shell 后已经自动激活环境，可以跳过。

### 3. 启动 vLLM 服务

在第一个终端执行：

```bash
bash scripts/run_vllm_server.sh
```

启动成功后通常会看到：

```text
[server] endpoint: http://127.0.0.1:8000/v1/completions
```

这个终端保持运行，不要关闭。

### 4. 跑单组 benchmark

另开一个终端：

```bash
cd /path/to/vllm-benchmark-suite
source ~/vllm312/bin/activate
bash scripts/run_benchmark.sh
```

### 5. 生成报告

```bash
bash scripts/render_report.sh
```

## 脚本说明

| 脚本 | 作用 | 典型用途 |
|---|---|---|
| `run_vllm_server.sh` | 启动本地 vLLM 服务 | 服务启动 |
| `run_benchmark.sh` | 跑一组 benchmark | 快速验证、单组复测 |
| `run_matrix.sh` | 跑矩阵 benchmark | 系统比较多组参数 |
| `render_report.sh` | 根据已有结果生成图表和报告 | 压测后整理结果 |

### `run_benchmark.sh`

单组测试：

```bash
bash scripts/run_benchmark.sh
```

临时覆盖参数：

```bash
bash scripts/run_benchmark.sh --concurrency 8 --max-tokens 64
```

### `run_matrix.sh`

矩阵测试：

```bash
bash scripts/run_matrix.sh
```

它会自动：
- 生成长 prompt 数据集
- 遍历矩阵参数组合
- 跳过部分明显过载组合
- 为每一组记录 summary、requests、GPU 日志

## 配置说明

配置文件：
[`configs/qwen25_1p5b_local.json`](/home/jeje/models/qwen25_1p5b_vllm_bench/configs/qwen25_1p5b_local.json)

### 默认服务配置

| 字段 | 默认值 |
|---|---|
| `host` | `127.0.0.1` |
| `port` | `8000` |
| `gpu_memory_utilization` | `0.75` |
| `max_model_len` | `2048` |
| `max_num_seqs` | `4` |

### 默认 benchmark 配置

| 字段 | 默认值 |
|---|---|
| `num_requests` | `100` |
| `concurrency` | `4` |
| `max_tokens` | `128` |
| `temperature` | `0.0` |

### 默认矩阵范围

| 维度 | 值 |
|---|---|
| prompt 档位 | `base / 256 / 1024` |
| concurrency | `1 / 2 / 4 / 8 / 12` |
| max_tokens | `64 / 128 / 256` |

说明：
- `prompt_tokens = 0` 在代码里表示 `base`，不是空 prompt
- `base` 使用 `prompts/sample_prompts.jsonl`
- `256 / 1024` 使用自动生成的长 prompt 数据集

## 输出结果

### 结果目录

| 路径 | 内容 |
|---|---|
| `results/logs/` | 服务日志、benchmark 日志、matrix 日志、GPU 采样 |
| `results/raw/` | `summary_*.json`、`raw_*.json` |
| `results/tables/` | `requests_*.csv` |
| `results/charts/` | 图表 PNG |
| `results/reports/` | HTML 报告和汇总 CSV |

### 重点文件

- 报告：[`results/reports/benchmark_report_latest.html`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/reports/benchmark_report_latest.html)
- 汇总表：[`results/reports/matrix_summary_latest.csv`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/reports/matrix_summary_latest.csv)

### 常见请求级字段

| 字段 | 含义 |
|---|---|
| `latency_ms` | 总延迟 |
| `ttft_ms` | 首 token 时间 |
| `prompt_tokens` | 输入 token 数 |
| `completion_tokens` | 输出 token 数 |
| `output_tokens_per_sec` | 输出 token 吞吐 |
| `total_tokens_per_sec` | 总 token 吞吐 |
| `status_code` | HTTP 状态码 |
| `error` | 错误信息 |

## 图表怎么看

### `benchmark_compare.png`

矩阵热力图，按 prompt 档位分层展示：
- `Req/s`
- `P50 TTFT`
- `P95 Latency`

适合快速回答：
- 哪组吞吐更高
- 哪组 TTFT 更低
- 哪组尾延迟更低

### `benchmark_prompt_trends.png`

趋势图，用来看：
- `base / 256 / 1024` 三类 prompt 在不同参数下如何变化
- 哪个 prompt 档位对 TTFT 或尾延迟更敏感

### `benchmark_pareto.png`

Pareto 图，用来看：
- 吞吐和尾延迟的平衡关系
- 越靠右表示吞吐越高
- 越靠下表示尾延迟越低
- 越靠右下通常越值得优先看

## 常见问题

### 为什么 `prompt_tokens=0` 不是空 prompt

这是代码里的标记值，表示使用 `base prompts`。
实际输入长度请看结果里的：
- `prompt_tokens`
- `avg_prompt_tokens`

### 为什么 `max_tokens` 越小，`req/s` 越大

这是正常现象。因为输出越短，请求越快结束，单位时间内完成的请求数就更多。

### 为什么某些高并发 / 长输出组合会失败

在小显存 GPU 上：
- 长 prompt
- 高并发
- 长输出
会同时抬高压力

所以脚本会跳过部分明显过载组合，但不保证所有组合都成功。

### 服务启动失败，提示显存不够怎么办

优先检查：
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_seqs`

### 50 系列为什么建议自己编译 vLLM

因为新架构 GPU 经常不能简单依赖预编译 wheel，源码编译通常更稳。

## 推荐工作流

### 快速验证

```bash
bash scripts/run_vllm_server.sh
bash scripts/run_benchmark.sh
bash scripts/render_report.sh
```

### 跑完整矩阵

```bash
bash scripts/run_vllm_server.sh
bash scripts/run_matrix.sh
bash scripts/render_report.sh
```

### 只复测一组参数

```bash
bash scripts/run_benchmark.sh --concurrency 8 --max-tokens 64
bash scripts/render_report.sh
```