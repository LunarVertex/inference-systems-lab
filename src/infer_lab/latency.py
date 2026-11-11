"""Abstract and profile-isolated step latency models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationProfile:
    """Explicitly sourced latency calibration parameters.

    Each hardware or literature source must use an independent profile to avoid
    extrapolating single-GPU RTX 4060 measurements to H100 or multi-GPU clusters.
    """

    name: str
    source_type: str  # e.g., "synthetic_baseline", "rtx4060", "paper_h100", "cluster_authorized"
    prefill_base_ms: float
    prefill_per_token_ms: float
    decode_base_ms: float
    decode_per_token_ms: float
    kv_per_token_ms: float = 0.0
    sched_overhead_base_ms: float = 0.0
    sched_overhead_per_req_ms: float = 0.0


class StepLatencyModel(ABC):
    """Interface for estimating forward-step and scheduling wall-clock time."""

    @abstractmethod
    def estimate_prefill(self, batch_size: int, prompt_tokens: int) -> float:
        """Estimate prefill compute duration in milliseconds."""

    @abstractmethod
    def estimate_decode(self, batch_size: int, kv_tokens: int) -> float:
        """Estimate decode step duration in milliseconds."""

    @abstractmethod
    def estimate_scheduling_overhead(self, num_requests: int) -> float:
        """Estimate CPU scheduler overhead in milliseconds."""


class LinearLatencyModel(StepLatencyModel):
    """Simple additive linear baseline parameterized by a CalibrationProfile."""

    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def estimate_prefill(self, batch_size: int, prompt_tokens: int) -> float:
        if prompt_tokens <= 0 or batch_size <= 0:
            return 0.0
        return self.profile.prefill_base_ms + self.profile.prefill_per_token_ms * prompt_tokens

    def estimate_decode(self, batch_size: int, kv_tokens: int) -> float:
        if batch_size <= 0:
            return 0.0
        return (
            self.profile.decode_base_ms
            + self.profile.decode_per_token_ms * batch_size
            + self.profile.kv_per_token_ms * max(0, kv_tokens)
        )

    def estimate_scheduling_overhead(self, num_requests: int) -> float:
        if num_requests <= 0:
            return 0.0
        return (
            self.profile.sched_overhead_base_ms
            + self.profile.sched_overhead_per_req_ms * num_requests
        )
