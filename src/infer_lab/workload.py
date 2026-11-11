"""Deterministic synthetic workload generators for controlled sensitivity tests."""

import random
from dataclasses import dataclass
from typing import Literal

from infer_lab.models import Request


@dataclass(frozen=True)
class SyntheticWorkloadConfig:
    """Parameters for generating deterministic synthetic request streams."""

    num_requests: int
    arrival_rate_rps: float
    arrival_mode: Literal["poisson", "constant"] = "poisson"
    length_mode: Literal["uniform", "bimodal"] = "uniform"
    short_prompt_range: tuple[int, int] = (64, 256)
    long_prompt_range: tuple[int, int] = (1024, 2048)
    long_request_ratio: float = 0.2
    output_range: tuple[int, int] = (16, 128)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.arrival_rate_rps <= 0:
            raise ValueError("arrival_rate_rps must be positive")
        if not 0.0 <= self.long_request_ratio <= 1.0:
            raise ValueError("long_request_ratio must be in [0.0, 1.0]")


def generate_synthetic_requests(config: SyntheticWorkloadConfig) -> list[Request]:
    """Generate a deterministic, arrival-sorted list of Requests."""
    rng = random.Random(config.seed)
    requests: list[Request] = []
    current_time_ms = 0.0
    mean_interarrival_ms = 1000.0 / config.arrival_rate_rps

    for idx in range(config.num_requests):
        if idx > 0:
            if config.arrival_mode == "constant":
                current_time_ms += mean_interarrival_ms
            else:
                # Exponential inter-arrival with rate = arrival_rate_rps / 1000 (per ms).
                current_time_ms += rng.expovariate(1.0 / mean_interarrival_ms)

        if config.length_mode == "bimodal" and rng.random() < config.long_request_ratio:
            p_min, p_max = config.long_prompt_range
        else:
            p_min, p_max = config.short_prompt_range

        prompt_tokens = rng.randint(p_min, p_max)
        max_output_tokens = rng.randint(config.output_range[0], config.output_range[1])
        requests.append(
            Request(
                request_id=f"req-{idx:05d}",
                arrival_time_ms=round(current_time_ms, 6),
                prompt_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
        )

    return requests
