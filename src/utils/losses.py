"""
Custom loss functions for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Loss(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Focal Loss down-weights easy examples and focuses training on hard negatives.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor in [0, 1] or tensor of class weights [num_classes].
               Can be used to balance class frequencies. Default: None (no weighting)
        gamma: Focusing parameter >= 0. Higher gamma focuses more on hard examples.
               gamma=0 reduces to standard cross entropy. Default: 2.0
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'. Default: 'mean'
        label_smoothing: Float in [0.0, 1.0]. Amount of label smoothing.
                        Default: 0.0 (no smoothing)
        normalize_by_alpha: If True, normalizes the loss by the sum of alpha weights.
                           This can help when using class weights to ensure the loss scale is consistent.

    Shape:
        - Input: (B, C) where B = batch size, C = number of classes (logits)
        - Target: (B,) class indices or (B, C) soft labels (for mixup)
        - Output: scalar if reduction='mean'/'sum', else (B,)

    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
        label_smoothing=0.0,
        normalize_by_alpha=False,
    ):
        super().__init__()

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.normalize_by_alpha = normalize_by_alpha
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0], got {label_smoothing}"
            )

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices or (B, C) soft labels (for mixup)

        Returns:
            Scalar loss if reduction='mean'/'sum', else (B,) losses
        """
        # Handle soft labels (from MixUp) vs hard labels
        if targets.ndim == 2:
            # Soft labels: targets is (B, C)
            # Apply label smoothing if specified
            if self.label_smoothing > 0:
                num_classes = targets.shape[1]
                targets = (
                    targets * (1 - self.label_smoothing)
                    + self.label_smoothing / num_classes
                )

            # Compute log probabilities
            log_probs = F.log_softmax(inputs, dim=1)

            # Cross entropy with soft labels
            ce_loss = -(targets * log_probs).sum(dim=1)

            # Compute probabilities for focal weight
            probs = torch.exp(log_probs)
            # Probability of the "correct" class (weighted by soft labels)
            p_t = (targets * probs).sum(dim=1)

            # Apply class weighting if alpha is provided
            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Weight by soft label distribution
                alpha_t = (self.alpha.unsqueeze(0) * targets).sum(dim=1)
            else:
                alpha_t = 1.0

        else:
            # Hard labels: targets is (B,)
            # Standard cross entropy
            ce_loss = F.cross_entropy(
                inputs,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )

            # Get probabilities
            probs = F.softmax(inputs, dim=1)
            # Probability of the true class
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Apply class weighting if alpha is provided
            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha[targets]
            else:
                alpha_t = 1.0

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Final loss: alpha * focal_weight * ce_loss
        loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            if self.alpha is not None and self.normalize_by_alpha:
                return (
                    loss.sum() / alpha_t.sum()
                )  # matches CrossEntropyLoss(weight=...), deviates from the focal loss paper
            else:
                return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
