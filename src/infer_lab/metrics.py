"""Performance metrics aggregation, percentile calculation, and SLO Goodput."""

import math
from dataclasses import dataclass
from statistics import mean, stdev

from infer_lab.models import Request, SimulationConfig

# Two-sided Student-t 95% critical values for df = 1..30.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def percentile(values: list[float], q: float) -> float:
    """Compute linear-interpolated percentile q in [0, 100]."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    if not 0.0 <= q <= 100.0:
        raise ValueError("Percentile q must be in [0, 100]")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_vals[lower]
    frac = pos - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, half_width_95_ci) using Student-t distribution."""
    if not values:
        raise ValueError("Cannot compute CI of an empty list")
    if len(values) == 1:
        return values[0], 0.0
    m = mean(values)
    s = stdev(values)
    df = len(values) - 1
    t_crit = _T_CRIT_95.get(df, 1.96)
    half_width = t_crit * (s / math.sqrt(len(values)))
    return m, half_width


@dataclass(frozen=True)
class RequestMetrics:
    """Per-request timing metrics extracted after completion."""

    request_id: str
    queue_delay_ms: float
    ttft_ms: float
    tpot_ms: float | None
    e2e_latency_ms: float
    slo_met: bool


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregated summary for a single DES simulation run."""

    num_requests: int
    tpot_sample_count: int
    total_duration_ms: float
    num_steps: int
    compute_tokens: int
    total_scheduling_overhead_ms: float
    slo_attainment_rate: float
    goodput_rps: float
    raw_output_tps: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    tpot_p99_ms: float | None
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    queue_delay_p50_ms: float
    queue_delay_p95_ms: float
    queue_delay_p99_ms: float


def extract_request_metrics(req: Request, sim_config: SimulationConfig) -> RequestMetrics:
    """Compute timing and SLO satisfaction for a single finished request."""
    if (
        req.first_scheduled_ms is None
        or req.first_token_ms is None
        or req.completion_ms is None
    ):
        raise ValueError(f"Request {req.request_id} is missing completion timestamps")

    queue_delay = req.first_scheduled_ms - req.arrival_time_ms
    ttft = req.first_token_ms - req.arrival_time_ms
    e2e = req.completion_ms - req.arrival_time_ms

    if req.max_output_tokens > 1:
        tpot: float | None = (req.completion_ms - req.first_token_ms) / (
            req.max_output_tokens - 1
        )
        tpot_ok = tpot <= sim_config.tpot_slo_ms
    else:
        tpot = None
        tpot_ok = True

    slo_met = (ttft <= sim_config.ttft_slo_ms) and tpot_ok
    return RequestMetrics(
        request_id=req.request_id,
        queue_delay_ms=queue_delay,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_latency_ms=e2e,
        slo_met=slo_met,
    )


def summarize_simulation(
    finished_requests: list[Request],
    sim_config: SimulationConfig,
    num_steps: int,
    compute_tokens: int,
    total_scheduling_overhead_ms: float,
) -> SimulationSummary:
    """Aggregate completed request metrics into percentiles and SLO Goodput.

    Total window duration is measured from the earliest request arrival to the
    latest request completion. Goodput counts only requests satisfying both
    TTFT and TPOT SLOs.
    """
    if not finished_requests:
        raise ValueError("Cannot summarize an empty list of finished requests")

    per_req = [extract_request_metrics(r, sim_config) for r in finished_requests]
    min_arrival = min(r.arrival_time_ms for r in finished_requests)
    max_completion = max(
        r.completion_ms for r in finished_requests if r.completion_ms is not None
    )
    total_duration_ms = max_completion - min_arrival
    if total_duration_ms <= 0:
        raise ValueError("Total simulation duration must be positive")
    total_duration_s = total_duration_ms / 1000.0

    ttfts = [m.ttft_ms for m in per_req]
    tpots = [m.tpot_ms for m in per_req if m.tpot_ms is not None]
    e2es = [m.e2e_latency_ms for m in per_req]
    queues = [m.queue_delay_ms for m in per_req]

    slo_met_count = sum(1 for m in per_req if m.slo_met)
    total_output_tokens = sum(r.max_output_tokens for r in finished_requests)

    return SimulationSummary(
        num_requests=len(finished_requests),
        tpot_sample_count=len(tpots),
        total_duration_ms=total_duration_ms,
        num_steps=num_steps,
        compute_tokens=compute_tokens,
        total_scheduling_overhead_ms=total_scheduling_overhead_ms,
        slo_attainment_rate=slo_met_count / len(finished_requests),
        goodput_rps=slo_met_count / total_duration_s,
        raw_output_tps=total_output_tokens / total_duration_s,
        ttft_p50_ms=percentile(ttfts, 50.0),
        ttft_p95_ms=percentile(ttfts, 95.0),
        ttft_p99_ms=percentile(ttfts, 99.0),
        tpot_p50_ms=percentile(tpots, 50.0) if tpots else None,
        tpot_p95_ms=percentile(tpots, 95.0) if tpots else None,
        tpot_p99_ms=percentile(tpots, 99.0) if tpots else None,
        e2e_p50_ms=percentile(e2es, 50.0),
        e2e_p95_ms=percentile(e2es, 95.0),
        e2e_p99_ms=percentile(e2es, 99.0),
        queue_delay_p50_ms=percentile(queues, 50.0),
        queue_delay_p95_ms=percentile(queues, 95.0),
        queue_delay_p99_ms=percentile(queues, 99.0),
    )
