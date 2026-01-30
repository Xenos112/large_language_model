"""
Implementation of Positional Embedding Layer
    - RoPE
    - FIX: I don't even know if this works🙂
"""

import torch
import torch.nn as nn

from src.config.config import ModelConfig
from src.utils.logger import Logger


class RoPE(nn.Module):
    def __init__(
        self,
        hidden_dim: int = ModelConfig.hidden_dim,
        max_sequence_length: int = ModelConfig.max_sequence_length,
        base: float = ModelConfig.base,
    ):
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.base = base
        self.logger = Logger(path="position_embedding.RoPE")  # See what to log later

        super().__init__()

        def build_cache(self, sequence_length: int):
            theta = 1.0 / (
                self.base
                ** (torch.arange(0, self.hidden_dim, 2).float() / self.hidden_dim)
            )

            sequence_index = torch.arange(sequence_length).float().unsqueeze(1)
            frequences = torch.outer(sequence_index, theta)

            self.register_buffer("cos_cached", torch.cos(frequences), persistant=False)
            self.register_buffer("sin_cached", torch.sin(frequences), persistant=False)

        def rotate_half(self, x: torch.Tensor):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]

            return torch.cat((-x2, x1), dim=-1)

        def forword(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            sequence_length: int,
            start_position: int = 0,
        ):
            if sequence_length > self.max_sequence_length:
                raise ValueError(
                    f"Sequence length {sequence_length} exceeds max sequence length {self.max_sequence_length}"
                )

            if start_position + sequence_length > self.cos_cached.shape[0]:
                self.build_cache(start_position + sequence_length)

            cos_cached = self.cos_cached[
                start_position : start_position + sequence_length
            ]
            sin_cached = self.sin_cached[
                start_position : start_position + sequence_length
            ]

            cos_cached = cos_cached.unsqueeze(0).unsqueeze(0)
            sin_cached = cos_cached.unsqueeze(0).unsqueeze(0)

            cos_cached = torch.repeat_interleave(cos_cached, 2, dim=-1)
            sin_cached = torch.repeat_interleave(sin_cached, 2, dim=-1)

            q_rotated = self.rotate_half(q)
            k_rotated = self.rotate_half(k)

            q_rotated = (q * cos_cached) + (self._rotate_half(q) * sin_cached)
            k_rotated = (k * cos_cached) + (self._rotate_half(k) * sin_cached)

            return q_rotated, k_rotated
