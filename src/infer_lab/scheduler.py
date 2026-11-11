"""Scheduler interface and Phase-1 FCFS + token-budget baseline."""

from abc import ABC, abstractmethod
from collections import deque

from infer_lab.models import Request, RequestStatus, SchedulerConfig, StepDecision


class BaseScheduler(ABC):
    """Abstract single-instance scheduler aligned with vLLM V1 step semantics."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Enqueue a newly arrived request."""

    @abstractmethod
    def schedule(self) -> StepDecision:
        """Select requests and token counts for the next execution step."""

    @abstractmethod
    def commit_step(
        self,
        decision: StepDecision,
        step_start_ms: float = 0.0,
        step_end_ms: float = 0.0,
    ) -> list[Request]:
        """Advance state after a step completes and return finished requests."""

    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Return True while waiting or running queues are non-empty."""


class FCFSScheduler(BaseScheduler):
    """Phase-1 FCFS scheduler inspired by `vllm/v1/core/sched/scheduler.py`.

    Running requests are scheduled first under a unified `token_budget`
    (`max_num_batched_tokens`), followed by waiting requests in FIFO order.
    KV block capacity constraints will be integrated in Week 3.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> StepDecision:
        decision = StepDecision()
        token_budget = self.config.max_num_batched_tokens

        # 1. Schedule already-running requests first (mirrors scheduler.py#L358-L492).
        for req in self.running:
            if token_budget <= 0:
                break
            tokens_needed = req.next_step_target_tokens
            if tokens_needed == 0:
                continue
            num_new_tokens = min(tokens_needed, token_budget)
            self._record_scheduled(req, num_new_tokens, decision)
            token_budget -= num_new_tokens

        # 2. Schedule waiting requests in FCFS order (mirrors scheduler.py#L541-L815).
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_seqs:
                break
            req = self.waiting[0]
            tokens_needed = req.next_step_target_tokens
            if not self.config.enable_chunked_prefill and tokens_needed > token_budget:
                break

            num_new_tokens = min(tokens_needed, token_budget)
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            self._record_scheduled(req, num_new_tokens, decision)
            token_budget -= num_new_tokens

        return decision

    @staticmethod
    def _record_scheduled(req: Request, num_new_tokens: int, decision: StepDecision) -> None:
        is_prefill = req.num_computed_tokens < req.prompt_tokens
        decision.scheduled_tokens[req.request_id] = num_new_tokens
        decision.total_scheduled_tokens += num_new_tokens
        if is_prefill:
            decision.prefill_tokens += num_new_tokens
            decision.prefill_batch_size += 1
        else:
            decision.decode_tokens += num_new_tokens
            decision.decode_batch_size += 1
            decision.decode_kv_tokens += req.num_computed_tokens

    def commit_step(
        self,
        decision: StepDecision,
        step_start_ms: float = 0.0,
        step_end_ms: float = 0.0,
    ) -> list[Request]:
        """Advance token progress after a simulated step and retire finished requests."""
        finished: list[Request] = []
        still_running: list[Request] = []
        for req in self.running:
            scheduled = decision.scheduled_tokens.get(req.request_id, 0)
            if scheduled > 0 and req.first_scheduled_ms is None:
                req.first_scheduled_ms = step_start_ms
            req.num_computed_tokens += scheduled
            if req.first_token_ms is None and req.num_computed_tokens >= req.prompt_tokens:
                req.first_token_ms = step_end_ms
            if req.num_computed_tokens >= req.total_target_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                req.completion_ms = step_end_ms
                finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        return finished
