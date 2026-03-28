# vLLM Benchmark Suite

这是一个本地 vLLM 性能测试小项目，用来评估模型在不同 prompt 长度、并发度、输出长度下的推理表现。

它适用于：
- Linux
- WSL2

项目当前默认模型路径是：
- `/home/jeje/models/Qwen2.5-1.5B-Instruct`

但整个项目并不依赖 Qwen，你也可以改成别的本地模型路径。

## 这个项目能做什么

支持以下能力：
- 启动本地 vLLM OpenAI 兼容服务
- 跑单组 benchmark
- 跑矩阵 benchmark
- 比较不同 prompt 长度、并发度、输出长度
- 记录 TTFT、延迟、吞吐、GPU 使用情况
- 生成热力图、趋势图、Pareto 图和 HTML 报告

## 目录结构

```text
.
├─ configs/
│  └─ qwen25_1p5b_local.json
├─ prompts/
│  ├─ sample_prompts.jsonl
│  └─ generated/
├─ results/
│  ├─ logs/
│  ├─ raw/
│  ├─ tables/
│  ├─ charts/
│  └─ reports/
├─ scripts/
│  ├─ run_vllm_server.sh
│  ├─ run_benchmark.sh
│  ├─ run_matrix.sh
│  └─ render_report.sh
└─ src/
   ├─ benchmark_client.py
   ├─ benchmark_common.py
   ├─ generate_long_prompts.py
   ├─ monitor_gpu.py
   └─ visualize_results.py
```

## 环境要求

### 操作系统

支持：
- Linux
- WSL2

### Python

建议：
- Python 3.12

### GPU

建议：
- NVIDIA GPU
- CUDA 环境可用
- `nvidia-smi` 可正常执行

### 基础依赖

当前脚本依赖这些 Python 包：
- `vllm`
- `transformers`
- `requests`
- `matplotlib`

你可以在虚拟环境中安装：

```bash
pip install vllm transformers requests matplotlib
```

## vLLM 安装说明

### 常规显卡

如果你的 GPU 和 CUDA 环境已经被官方 wheel 支持，可以直接安装：

```bash
pip install vllm
```

### RTX 50 系列 / 新架构显卡

如果你使用的是 50 系列显卡，通常不能完全依赖现成 wheel，建议你自行编译 vLLM。

也就是说：
- 50 系列用户建议使用源码编译
- 需要确保本机的 CUDA、PyTorch、编译环境与目标架构匹配
- 编译成功后再运行本项目

这类环境下，README 只负责说明 benchmark 项目如何使用，不负责替代 vLLM 编译文档。

如果你已经能正常执行类似命令：

```bash
python -m vllm.entrypoints.openai.api_server --help
```

说明 vLLM 基本已经可用了。

## 推荐运行环境

建议使用单独虚拟环境，例如：

```bash
python3.12 -m venv ~/vllm312
source ~/vllm312/bin/activate
pip install --upgrade pip
pip install vllm transformers requests matplotlib
```

如果你已经有可用环境，例如 `vllm312`，直接激活即可。

## 配置文件

配置文件位置：

[`configs/qwen25_1p5b_local.json`](/home/jeje/models/qwen25_1p5b_vllm_bench/configs/qwen25_1p5b_local.json)

默认关键配置：
- 模型路径：`/home/jeje/models/Qwen2.5-1.5B-Instruct`
- 服务端口：`127.0.0.1:8000`
- `gpu_memory_utilization = 0.75`
- `max_model_len = 2048`
- `max_num_seqs = 4`
- `num_requests = 100`

矩阵测试默认覆盖：
- prompt 档位：`base / 256 / 1024`
- 并发：`1 / 2 / 4 / 8 / 12`
- 输出长度：`64 / 128 / 256`

说明：
- `prompt_tokens = 0` 在代码里表示 `base`，不是空 prompt
- `base` 使用 `prompts/sample_prompts.jsonl`
- `256 / 1024` 使用自动生成的长 prompt 数据集

## 快速启动

### 1. 进入项目目录

```bash
cd /path/to/vllm-benchmark-suite
```

如果你是当前这份本地目录，就是：

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

成功后会看到类似：

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

### `run_vllm_server.sh`

作用：
- 根据配置启动本地 vLLM OpenAI 服务
- 服务日志写入 `results/logs/server_*.log`

用法：

```bash
bash scripts/run_vllm_server.sh
```

### `run_benchmark.sh`

作用：
- 跑一组 benchmark
- 记录请求级明细
- 记录 GPU 采样日志

用法：

```bash
bash scripts/run_benchmark.sh
```

临时覆盖参数示例：

```bash
bash scripts/run_benchmark.sh --concurrency 8 --max-tokens 64
```

### `run_matrix.sh`

作用：
- 自动生成长 prompt 数据集
- 遍历矩阵参数组合
- 自动跳过部分明显过载组合
- 为每一组保存 summary、requests、GPU 日志

用法：

```bash
bash scripts/run_matrix.sh
```

### `render_report.sh`

作用：
- 根据已有结果重新生成图表和 HTML 报告
- 不会重新压测

用法：

```bash
bash scripts/render_report.sh
```

## 输出结果说明

### 日志

目录：
[`results/logs`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/logs)

包括：
- `server_*.log`
- `benchmark_*.log`
- `matrix_*.log`
- `render_*.log`
- `gpu_metrics_*.csv`

### 原始结果

目录：
[`results/raw`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/raw)

包括：
- `summary_*.json`
- `raw_*.json`

### 请求级明细

目录：
[`results/tables`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/tables)

包括：
- `requests_*.csv`

常见字段：
- `success`
- `status_code`
- `latency_ms`
- `ttft_ms`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `output_tokens_per_sec`
- `total_tokens_per_sec`
- `error`

### 图表

目录：
[`results/charts`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/charts)

当前主要图表：
- `benchmark_compare.png`：矩阵热力图
- `benchmark_prompt_trends.png`：prompt 分层趋势图
- `benchmark_pareto.png`：吞吐和尾延迟权衡图

### 报告

目录：
[`results/reports`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/reports)

主要文件：
- `benchmark_report_latest.html`
- `matrix_summary_latest.csv`

推荐直接打开：
[`results/reports/benchmark_report_latest.html`](/home/jeje/models/qwen25_1p5b_vllm_bench/results/reports/benchmark_report_latest.html)

## 图表怎么看

### `benchmark_compare.png`

这是矩阵热力图。

按 prompt 档位分层展示：
- `Req/s`
- `P50 TTFT`
- `P95 Latency`

适合快速找：
- 吞吐更好的组合
- TTFT 更低的组合
- 尾延迟更低的组合

### `benchmark_prompt_trends.png`

这是趋势图。

用途：
- 比较 `base / 256 / 1024` 三类 prompt 在不同参数下的变化趋势
- 看不同 prompt 长度对 TTFT、Req/s、P95 的整体影响

### `benchmark_pareto.png`

这是权衡图。

用途：
- 看吞吐和尾延迟的平衡关系
- 越靠右说明吞吐越高
- 越靠下说明尾延迟越低
- 越靠右下通常越值得重点关注

## 关键指标解释

- `success_rate`：请求成功率
- `avg_latency_ms`：平均总延迟
- `p50_latency_ms`：中位延迟
- `p95_latency_ms`：尾延迟
- `avg_ttft_ms` / `p50_ttft_ms`：首 token 时间
- `avg_output_tokens_per_sec`：输出 token 吞吐
- `avg_total_tokens_per_sec`：总 token 吞吐
- `effective_requests_per_sec`：整体请求吞吐

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

## 常见问题

### 1. 服务启动失败，提示显存不够

优先检查：
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_seqs`

### 2. 为什么 `prompt_tokens=0` 不是空 prompt

这是代码里的标记值，表示使用 `base prompts`。
实际输入长度请看结果里的：
- `prompt_tokens`
- `avg_prompt_tokens`

### 3. 为什么 `max_tokens` 越小，`req/s` 越大

这是正常现象。
因为输出越短，请求越快结束，单位时间能完成的请求数更多。

### 4. 为什么某些高并发 / 长输出组合会失败

因为在小显存 GPU 上：
- 长 prompt
- 高并发
- 长输出
会同时抬高压力。

### 5. 50 系列为什么要自己编译

因为新架构 GPU 往往不能简单依赖预编译 wheel。
如果你的环境不在官方 wheel 的稳定支持范围内，源码编译通常更稳妥。

本项目默认前提是：
- 你的 vLLM 已经在当前环境中可用
- benchmark 项目只负责测试，不负责替代 vLLM 自身的安装/编译文档