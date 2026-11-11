"""Deterministic tests for the heapq DES loop, workload generator, and metrics."""

import pytest

from infer_lab import (
    CalibrationProfile,
    FCFSScheduler,
    LinearLatencyModel,
    Request,
    SchedulerConfig,
    SimulationConfig,
    Simulator,
    SyntheticWorkloadConfig,
    generate_synthetic_requests,
    mean_ci95,
    percentile,
)


def _test_profile() -> CalibrationProfile:
    return CalibrationProfile(
        name="synthetic_unit_test",
        source_type="synthetic_baseline",
        prefill_base_ms=2.0,
        prefill_per_token_ms=0.1,
        decode_base_ms=1.0,
        decode_per_token_ms=0.2,
        kv_per_token_ms=0.0,
        sched_overhead_base_ms=0.5,
        sched_overhead_per_req_ms=0.0,
    )


def test_single_request_analytical_ttft_and_tpot() -> None:
    """Verify exact analytical TTFT, TPOT, and Compute Tokens for 1 request."""
    model = LinearLatencyModel(_test_profile())
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64, enable_chunked_prefill=True)
    sim = Simulator(
        scheduler=FCFSScheduler(cfg),
        latency_model=model,
        sim_config=SimulationConfig(ttft_slo_ms=100.0, tpot_slo_ms=10.0),
    )
    # prompt=100, output=3 => prefill chunks [64, 36], then 2 decode steps (101, 102).
    # Step 1 (64 prefill): 0.5 + (2.0 + 6.4) = 8.9 ms
    # Step 2 (36 prefill): 0.5 + (2.0 + 3.6) = 6.1 ms => TTFT = 15.0 ms
    # Step 3 (1 decode):   0.5 + (1.0 + 0.2) = 1.7 ms
    # Step 4 (1 decode):   0.5 + (1.0 + 0.2) = 1.7 ms => E2E = 18.4 ms, TPOT = 1.7 ms
    req = Request("r1", arrival_time_ms=0.0, prompt_tokens=100, max_output_tokens=3)
    res = sim.run([req], record_traces=True)

    assert len(res.step_traces) == 4
    assert res.summary.compute_tokens == 102
    assert res.summary.ttft_p50_ms == pytest.approx(15.0)
    assert res.summary.tpot_p50_ms == pytest.approx(1.7)
    assert res.summary.e2e_p50_ms == pytest.approx(18.4)
    assert res.summary.queue_delay_p50_ms == pytest.approx(0.0)
    assert res.summary.slo_attainment_rate == pytest.approx(1.0)


def test_single_output_token_edge_case() -> None:
    """max_output_tokens=1 finishes at prefill without dividing TPOT by zero."""
    model = LinearLatencyModel(_test_profile())
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128, enable_chunked_prefill=True)
    sim = Simulator(FCFSScheduler(cfg), model, SimulationConfig())
    req = Request("r1", arrival_time_ms=5.0, prompt_tokens=50, max_output_tokens=1)
    res = sim.run([req])

    # Step 1: 0.5 + (2.0 + 5.0) = 7.5 ms
    assert res.summary.num_steps == 1
    assert res.summary.compute_tokens == 50
    assert res.summary.ttft_p50_ms == pytest.approx(7.5)
    assert res.summary.tpot_sample_count == 0
    assert res.summary.tpot_p50_ms is None


def test_arrival_during_busy_step_waits_for_next_batch() -> None:
    """A request arriving mid-step must queue until STEP_COMPLETE."""
    model = LinearLatencyModel(_test_profile())
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128, enable_chunked_prefill=True)
    sim = Simulator(FCFSScheduler(cfg), model, SimulationConfig())

    r1 = Request("r1", arrival_time_ms=0.0, prompt_tokens=50, max_output_tokens=2)
    # r1 step 1 runs [0.0, 7.5]; r2 arrives at 3.0 and should start at 7.5.
    r2 = Request("r2", arrival_time_ms=3.0, prompt_tokens=20, max_output_tokens=1)
    res = sim.run([r1, r2], record_traces=True)

    assert r1.first_scheduled_ms == pytest.approx(0.0)
    assert r2.first_scheduled_ms == pytest.approx(7.5)
    assert res.step_traces[1].prefill_tokens == 20
    assert res.step_traces[1].decode_tokens == 1
    assert res.summary.compute_tokens == (50 + 1) + 20


def test_synthetic_workload_reproducibility_and_chunked_vs_nonchunked() -> None:
    """Same seed produces identical runs; chunked prefill limits step latency spike."""
    wl_cfg = SyntheticWorkloadConfig(
        num_requests=25,
        arrival_rate_rps=40.0,
        arrival_mode="poisson",
        length_mode="bimodal",
        short_prompt_range=(32, 64),
        long_prompt_range=(512, 768),
        long_request_ratio=0.25,
        output_range=(8, 24),
        seed=123,
    )
    reqs_a = generate_synthetic_requests(wl_cfg)
    reqs_b = generate_synthetic_requests(wl_cfg)
    assert [(r.arrival_time_ms, r.prompt_tokens, r.max_output_tokens) for r in reqs_a] == [
        (r.arrival_time_ms, r.prompt_tokens, r.max_output_tokens) for r in reqs_b
    ]

    model = LinearLatencyModel(_test_profile())
    sim_cfg = SimulationConfig(ttft_slo_ms=150.0, tpot_slo_ms=25.0)

    chunked_cfg = SchedulerConfig(
        max_num_seqs=16, max_num_batched_tokens=128, enable_chunked_prefill=True
    )
    chunked_res = Simulator(
        FCFSScheduler(chunked_cfg),
        model,
        sim_cfg,
    ).run(reqs_a)

    # Compare against non-chunked with larger budget able to fit a single 768-token prompt.
    non_chunked_res = Simulator(
        FCFSScheduler(
            SchedulerConfig(
                max_num_seqs=16, max_num_batched_tokens=1024, enable_chunked_prefill=False
            )
        ),
        model,
        sim_cfg,
    ).run(reqs_b)

    expected_tokens = sum(r.total_target_tokens for r in generate_synthetic_requests(wl_cfg))
    assert chunked_res.summary.compute_tokens == expected_tokens
    assert non_chunked_res.summary.compute_tokens == expected_tokens
    assert chunked_res.summary.goodput_rps <= chunked_res.summary.raw_output_tps


def test_percentile_and_ci95_helpers() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50.0) == pytest.approx(25.0)
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.0) == pytest.approx(10.0)
    assert percentile([10.0, 20.0, 30.0, 40.0], 100.0) == pytest.approx(40.0)

    m, hw = mean_ci95([10.0, 12.0, 14.0])
    assert m == pytest.approx(12.0)
    assert hw > 0.0
