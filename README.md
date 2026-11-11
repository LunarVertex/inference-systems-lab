# inference-systems-lab

A **vLLM V1-inspired Discrete Event Simulator** for reproducible LLM inference scheduling and KV cache research.

## Scope & Integrity Notice

- **Design Position**: This simulator is *inspired by* official vLLM V1 scheduling primitives (`vllm/v1/core/sched/scheduler.py`), not a full reproduction of production vLLM V1.
- **Reference Baseline**: `vllm-project/vllm` tag `v0.17.0` (commit `b31e9326a7d9394aab8c767f8ebe225c65594b60`).
- **Cleanroom Boundary**: Only synthetic workloads, public traces, and explicitly decoupled calibration profiles belong in this repository. Never commit internal cluster topologies, proprietary configs, or non-public traces.

## Phase 1 Roadmap (12h/month budget)

- **Week 1**: Official vLLM `v0.17.0` scheduler anchor analysis, repository scaffolding, deterministic invariant tests.
- **Week 2**: Discrete event loop (`heapq`), request lifecycle, FCFS baseline, synthetic workload generator.
- **Week 3**: Token budget & chunked prefill mechanics, KV block accounting & preemption, public trace slice replay.
- **Week 4**: First controlled ablation (`FCFS` vs `Chunked Prefill`) with P99/Goodput/Compute Tokens and falsifiable counter-example analysis.

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```
