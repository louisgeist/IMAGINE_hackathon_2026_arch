from torchvision.ops import sigmoid_focal_loss
from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        reduction: str,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            source, target, self.alpha, self.gamma, self.reduction
        )
