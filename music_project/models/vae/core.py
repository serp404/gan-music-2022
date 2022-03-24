import torch.nn as nn
from torch import Tensor


class MelVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

    def encode(x: Tensor) -> Tensor:
        return x

    def decode(z: Tensor) -> Tensor:
        return z

    def forward(x: Tensor) -> Tensor:
        return x

    def sample() -> Tensor:
        return 0.
