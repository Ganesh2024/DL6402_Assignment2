"""Reusable custom layers for the perception pipeline."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Hand-rolled dropout that mirrors the inverted-dropout convention used in
    modern deep learning frameworks, but built entirely from primitive tensor ops.

    During training a random binary mask is sampled from a Bernoulli distribution
    with keep-probability (1 - p).  Each surviving activation is scaled up by
    1/(1-p) so that the expected sum of activations stays constant — this means
    the forward pass at inference time requires zero extra work (no re-scaling).

    When self.training is False the layer acts as a pure identity mapping.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of zeroing out any individual element.
               Must satisfy 0 <= p < 1.
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), received {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No-op during evaluation — standard behaviour
        if not self.training:
            return x

        # Edge case: p == 0 means keep everything
        if self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p

        # Sample a Bernoulli mask with the same shape and device as the input.
        # torch.empty then bernoulli_ keeps the operation in-graph for autograd.
        mask = torch.empty_like(x).bernoulli_(keep_prob)

        # Inverted scaling: divide by keep probability so test-time is a no-op
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
