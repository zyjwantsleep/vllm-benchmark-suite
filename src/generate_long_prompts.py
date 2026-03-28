from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PARAGRAPHS = [
    "背景材料一：vLLM 通过 continuous batching 将不同请求的 decode 阶段动态合批，从而减少 GPU 空转时间。在服务负载较高时，这种机制通常能明显提升总体吞吐，但也会改变单请求的排队时间分布。",
    "背景材料二：在大模型推理中，prefill 主要处理输入 prompt，decode 主要逐 token 生成输出。长 prompt 会显著抬高 prefill 成本，因此常常导致 TTFT 上升；而长输出则更明显影响整个请求的完成时延。",
    "背景材料三：线上推理服务不能只看平均延迟，因为尾延迟会直接影响用户体验。P50 能反映中位体验，P95 或 P99 更适合观察抖动、排队和极端慢请求。",
    "背景材料四：在显存较小的 GPU 上部署模型时，常见的调优参数包括 gpu_memory_utilization、max_model_len、max_num_seqs 和并发度。如果这些参数设置过高，服务可能在启动阶段或高压场景下失败。",
    "背景材料五：GPU 利用率高并不一定意味着系统健康。若 TTFT 很高、尾延迟恶化、请求成功率下降，说明系统可能已经进入拥塞状态。此时仅追求更高吞吐并不一定符合业务目标。",
    "背景材料六：流式输出适合聊天场景，因为用户能更早看到首个 token。对于同样的请求，TTFT 通常比完整响应时间更能代表主观交互体验。",
    "背景材料七：压测时应分别控制输入长度、输出长度和并发度。若把这些变量同时大幅提高，得到的结果很难定位瓶颈究竟来自 prefill、decode 还是调度。",
    "背景材料八：真实服务中的请求分布通常是混合的，既有短问题，也有长上下文整理任务，还有代码生成、结构化输出和知识问答。因此高质量 benchmark 不应只使用少量高度相似的 prompt。",
    "背景材料九：监控数据建议同时包括请求级日志、汇总指标和 GPU 采样。这样在出现异常时，既能看到总体趋势，也能回查具体是哪类请求、哪组参数导致问题。",
    "背景材料十：如果并发继续增加而 req/s 提升有限，同时 P95 与 TTFT 快速上升，通常说明已经接近该机器的有效上限。此时更适合降低并发或缩短输出长度，而不是继续堆压。",
]

QUESTIONS = [
    "问题：请按要点总结上述材料，并解释为什么 TTFT 和总体吞吐不一定同步变化。",
    "问题：请基于上述材料，总结在 8GB 显存 GPU 上部署小模型时最关键的三个调优方向。",
    "问题：请根据上述信息，比较长 prompt、长输出和高并发三者对线上体验的不同影响。",
    "问题：请结合材料，说明为什么压测时必须同时记录成功率、TTFT、P95 和 GPU 利用率。",
]


def build_prompt(tokenizer: AutoTokenizer, target_tokens: int, seed: int) -> str:
    prefix = (
        "请阅读以下多段中文技术背景资料，并根据最后的问题给出结构化回答。"
        "回答要求：使用中文、分点作答、不要省略关键指标。\n\n"
    )
    suffix = "\n\n输出格式：先给结论，再给原因，最后给调优建议。"

    segments: list[str] = []
    index = seed % len(PARAGRAPHS)
    question = QUESTIONS[seed % len(QUESTIONS)]

    while True:
        segments.append(PARAGRAPHS[index])
        index = (index + 1) % len(PARAGRAPHS)
        body = "\n\n".join(segments)
        text = prefix + body + "\n\n" + question + suffix
        if len(tokenizer.encode(text, add_special_tokens=False)) >= target_tokens:
            token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
            return tokenizer.decode(token_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long prompt datasets.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", type=int, required=True)
    parser.add_argument("--samples-per-target", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for target in args.targets:
        output_path = args.output_dir / f"long_prompts_{target}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for index in range(args.samples_per_target):
                row = {
                    "id": f"long_{target}_{index + 1:02d}",
                    "prompt": build_prompt(tokenizer, target, index),
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(output_path)


if __name__ == "__main__":
    main()