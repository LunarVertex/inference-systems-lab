"""Deterministic invariant tests for Phase-1 Week 1 skeleton."""

import pytest

from infer_lab import (
    CalibrationProfile,
    FCFSScheduler,
    LinearLatencyModel,
    Request,
    RequestStatus,
    SchedulerConfig,
)


def test_request_validation_and_token_targets() -> None:
    req = Request(request_id="r1", arrival_time_ms=0.0, prompt_tokens=100, max_output_tokens=10)
    assert req.total_target_tokens == 109
    assert req.next_step_target_tokens == 100

    req.num_computed_tokens = 100
    assert req.next_step_target_tokens == 1

    req.num_computed_tokens = 109
    assert req.next_step_target_tokens == 0

    single_tok = Request(
        request_id="r2", arrival_time_ms=0.0, prompt_tokens=50, max_output_tokens=1
    )
    assert single_tok.total_target_tokens == 50

    with pytest.raises(ValueError):
        Request(request_id="bad", arrival_time_ms=0.0, prompt_tokens=0, max_output_tokens=5)


def test_token_budget_never_exceeded_and_chunked_prefill_respected() -> None:
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64, enable_chunked_prefill=True)
    sched = FCFSScheduler(cfg)
    sched.add_request(Request("r1", arrival_time_ms=0.0, prompt_tokens=100, max_output_tokens=2))
    sched.add_request(Request("r2", arrival_time_ms=1.0, prompt_tokens=20, max_output_tokens=2))

    # Step 1: r1 gets chunked to 64 tokens, r2 remains waiting.
    d1 = sched.schedule()
    assert d1.total_scheduled_tokens == 64
    assert d1.scheduled_tokens == {"r1": 64}
    sched.commit_step(d1)
    assert len(sched.running) == 1
    assert len(sched.waiting) == 1

    # Step 2: r1 finishes remaining 36 prompt tokens; r2 enters with remaining 20 tokens.
    d2 = sched.schedule()
    assert d2.total_scheduled_tokens == 56
    assert d2.scheduled_tokens == {"r1": 36, "r2": 20}
    sched.commit_step(d2)
    assert len(sched.running) == 2


def test_non_chunked_prefill_blocks_oversized_waiting_request() -> None:
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64, enable_chunked_prefill=False)
    sched = FCFSScheduler(cfg)
    sched.add_request(Request("r1", arrival_time_ms=0.0, prompt_tokens=50, max_output_tokens=2))
    sched.add_request(Request("r2", arrival_time_ms=1.0, prompt_tokens=40, max_output_tokens=2))

    d1 = sched.schedule()
    # r1 fits (50 <= 64); r2 needs 40 > remaining 14, so non-chunked mode stops.
    assert d1.scheduled_tokens == {"r1": 50}
    sched.commit_step(d1)
    assert len(sched.waiting) == 1


def test_token_conservation_until_completion() -> None:
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=32, enable_chunked_prefill=True)
    sched = FCFSScheduler(cfg)
    reqs = [
        Request("r1", arrival_time_ms=0.0, prompt_tokens=45, max_output_tokens=3),
        Request("r2", arrival_time_ms=0.0, prompt_tokens=20, max_output_tokens=2),
    ]
    for r in reqs:
        sched.add_request(r)

    total_scheduled = 0
    steps = 0
    while sched.has_unfinished_requests():
        d = sched.schedule()
        assert 0 < d.total_scheduled_tokens <= cfg.max_num_batched_tokens
        total_scheduled += d.total_scheduled_tokens
        sched.commit_step(d)
        steps += 1
        assert steps < 100

    assert total_scheduled == sum(r.total_target_tokens for r in reqs)
    assert all(r.status == RequestStatus.FINISHED_LENGTH_CAPPED for r in reqs)


def test_latency_profile_isolation() -> None:
    profile = CalibrationProfile(
        name="synthetic_unit_test",
        source_type="synthetic_baseline",
        prefill_base_ms=2.0,
        prefill_per_token_ms=0.05,
        decode_base_ms=1.0,
        decode_per_token_ms=0.1,
        kv_per_token_ms=0.001,
        sched_overhead_base_ms=0.2,
        sched_overhead_per_req_ms=0.01,
    )
    model = LinearLatencyModel(profile)
    assert model.estimate_prefill(1, 100) == pytest.approx(7.0)
    assert model.estimate_decode(4, 1000) == pytest.approx(2.4)
    assert model.estimate_scheduling_overhead(10) == pytest.approx(0.3)
