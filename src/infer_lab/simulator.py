"""Heapq-based discrete event simulator for single-instance LLM scheduling."""

import heapq
from dataclasses import dataclass
from enum import Enum, auto

from infer_lab.latency import StepLatencyModel
from infer_lab.metrics import SimulationSummary, summarize_simulation
from infer_lab.models import Request, SimulationConfig, StepDecision
from infer_lab.scheduler import BaseScheduler


class EventType(Enum):
    """Discrete simulation event types."""

    ARRIVAL = auto()
    STEP_COMPLETE = auto()


@dataclass(frozen=True)
class StepTrace:
    """Lightweight per-step execution record for debugging and invariant checks."""

    step_index: int
    start_ms: float
    end_ms: float
    scheduling_overhead_ms: float
    prefill_tokens: int
    decode_tokens: int
    total_scheduled_tokens: int


@dataclass
class SimulationResult:
    """Complete output of a single DES run."""

    finished_requests: list[Request]
    summary: SimulationSummary
    step_traces: list[StepTrace]


class Simulator:
    """vLLM V1-inspired single-instance Discrete Event Simulator.

    Explicit Week-2 assumptions:
    1. Infinite KV cache capacity (KV Block Manager and preemption arrive in Week 3).
    2. Additive step duration: scheduling overhead + prefill + decode latency.
    3. Arrivals during an active GPU step wait until STEP_COMPLETE to be batched.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        latency_model: StepLatencyModel,
        sim_config: SimulationConfig | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.latency_model = latency_model
        self.sim_config = sim_config or SimulationConfig()

    def run(
        self,
        requests: list[Request],
        record_traces: bool = False,
    ) -> SimulationResult:
        """Execute deterministic event-driven simulation until all requests finish."""
        if not requests:
            raise ValueError("requests must not be empty")

        event_heap: list[tuple[float, int, EventType, object]] = []
        seq = 0
        for req in sorted(requests, key=lambda r: r.arrival_time_ms):
            heapq.heappush(event_heap, (req.arrival_time_ms, seq, EventType.ARRIVAL, req))
            seq += 1

        finished_requests: list[Request] = []
        step_traces: list[StepTrace] = []
        engine_busy = False
        active_decision: StepDecision | None = None
        active_step_start_ms = 0.0
        num_steps = 0
        compute_tokens = 0
        total_sched_overhead_ms = 0.0
        last_time_ms = 0.0

        while event_heap:
            current_time_ms = event_heap[0][0]
            if current_time_ms < last_time_ms:
                raise AssertionError("DES clock moved backwards")
            last_time_ms = current_time_ms

            # Drain all events occurring at the exact same timestamp before scheduling.
            while event_heap and event_heap[0][0] == current_time_ms:
                _, _, event_type, payload = heapq.heappop(event_heap)
                if event_type is EventType.ARRIVAL:
                    assert isinstance(payload, Request)
                    self.scheduler.add_request(payload)
                elif event_type is EventType.STEP_COMPLETE:
                    assert active_decision is not None
                    finished = self.scheduler.commit_step(
                        active_decision,
                        step_start_ms=active_step_start_ms,
                        step_end_ms=current_time_ms,
                    )
                    finished_requests.extend(finished)
                    engine_busy = False
                    active_decision = None

            if not engine_busy and self.scheduler.has_unfinished_requests():
                decision = self.scheduler.schedule()
                if decision.total_scheduled_tokens <= 0:
                    raise RuntimeError("Scheduler returned empty step with unfinished requests")

                sched_ms = self.latency_model.estimate_scheduling_overhead(
                    len(decision.scheduled_tokens)
                )
                prefill_ms = self.latency_model.estimate_prefill(
                    decision.prefill_batch_size, decision.prefill_tokens
                )
                decode_ms = self.latency_model.estimate_decode(
                    decision.decode_batch_size, decision.decode_kv_tokens
                )
                step_dur_ms = sched_ms + prefill_ms + decode_ms
                if step_dur_ms <= 0:
                    raise ValueError("Step duration must be strictly positive")

                step_end_ms = current_time_ms + step_dur_ms
                active_decision = decision
                active_step_start_ms = current_time_ms
                engine_busy = True

                num_steps += 1
                compute_tokens += decision.total_scheduled_tokens
                total_sched_overhead_ms += sched_ms

                if record_traces:
                    step_traces.append(
                        StepTrace(
                            step_index=num_steps,
                            start_ms=current_time_ms,
                            end_ms=step_end_ms,
                            scheduling_overhead_ms=sched_ms,
                            prefill_tokens=decision.prefill_tokens,
                            decode_tokens=decision.decode_tokens,
                            total_scheduled_tokens=decision.total_scheduled_tokens,
                        )
                    )

                heapq.heappush(
                    event_heap, (step_end_ms, seq, EventType.STEP_COMPLETE, None)
                )
                seq += 1

        summary = summarize_simulation(
            finished_requests=finished_requests,
            sim_config=self.sim_config,
            num_steps=num_steps,
            compute_tokens=compute_tokens,
            total_scheduling_overhead_ms=total_sched_overhead_ms,
        )
        return SimulationResult(
            finished_requests=finished_requests,
            summary=summary,
            step_traces=step_traces,
        )
