"""Core request, state, and scheduler configuration models."""

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    """Minimal request lifecycle aligned with vLLM v0.17.0 RequestStatus."""

    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()

    def is_finished(self) -> bool:
        """Return True if the request has reached a terminal state."""
        return self in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        )


@dataclass
class Request:
    """Single LLM inference request tracked by the DES scheduler.

    Follows the unified token-progress model of vLLM V1 (`vllm/v1/request.py`):
    instead of separate prefill/decode states, progress is measured by
    `num_computed_tokens` catching up toward `prompt_tokens + max_output_tokens`.
    """

    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    max_output_tokens: int
    prefix_id: str | None = None
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0
    num_preemptions: int = 0
    first_scheduled_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be non-negative")

    @property
    def total_target_tokens(self) -> int:
        """Total token computations required (prompt + remaining decode steps).

        In autoregressive LLM serving, computing the final prompt token yields
        the first output token, leaving `max_output_tokens - 1` decode steps.
        """
        return self.prompt_tokens + self.max_output_tokens - 1

    @property
    def remaining_prompt_tokens(self) -> int:
        """Prompt tokens not yet computed."""
        return max(0, self.prompt_tokens - self.num_computed_tokens)

    @property
    def next_step_target_tokens(self) -> int:
        """Tokens currently ready to compute in the next forward step.

        Before prefill finishes, all remaining prompt tokens are targetable;
        once prompt is computed, at most 1 decode token is generated per step.
        """
        if self.num_computed_tokens < self.prompt_tokens:
            return self.prompt_tokens - self.num_computed_tokens
        if self.num_computed_tokens < self.total_target_tokens:
            return 1
        return 0


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler capacity and chunking parameters."""

    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = True
    vllm_reference_tag: str = "v0.17.0"
    vllm_reference_sha: str = "b31e9326a7d9394aab8c767f8ebe225c65594b60"

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a single-instance DES run."""

    ttft_slo_ms: float = 500.0
    tpot_slo_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.ttft_slo_ms <= 0 or self.tpot_slo_ms <= 0:
            raise ValueError("SLO thresholds must be positive")


@dataclass
class StepDecision:
    """Output of a single scheduling step."""

    scheduled_tokens: dict[str, int] = field(default_factory=dict)
    preempted_req_ids: list[str] = field(default_factory=list)
    total_scheduled_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_batch_size: int = 0
    decode_batch_size: int = 0
    decode_kv_tokens: int = 0
