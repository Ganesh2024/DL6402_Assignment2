"""Custom Intersection-over-Union loss for bounding-box regression.

Boxes are expected in (x_center, y_center, width, height) pixel-space format,
which matches the output convention of VGG11Localizer.

Loss value = 1 - IoU, so the range is always [0, 1]:
  * perfect overlap  → IoU = 1 → loss = 0
  * no overlap at all → IoU = 0 → loss = 1
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Differentiable IoU loss.

    Supports three reduction modes:
      'mean'  – average over the batch (default, required by spec)
      'sum'   – sum over the batch
      'none'  – return per-sample losses without reducing

    A small epsilon guards against division-by-zero when a predicted or
    ground-truth box has near-zero area.
    """

    _VALID_REDUCTIONS = {"mean", "sum", "none"}

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    # ------------------------------------------------------------------
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) → (x1, y1, x2, y2) corner format."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes   : [B, 4] predicted   (cx, cy, w, h) in pixel space
            target_boxes : [B, 4] ground-truth (cx, cy, w, h) in pixel space

        Returns:
            Scalar loss (or [B] tensor when reduction='none').
        """
        pred_xyxy   = self._cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self._cxcywh_to_xyxy(target_boxes)

        # Coordinates of the intersection rectangle
        inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        # Clamp to zero: if boxes don't overlap the intersection is empty
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        intersection = inter_w * inter_h

        # Individual box areas
        pred_area   = pred_boxes[:, 2]   * pred_boxes[:, 3]        # w * h
        target_area = target_boxes[:, 2] * target_boxes[:, 3]

        union = pred_area + target_area - intersection + self.eps

        iou_per_sample = intersection / union                       # [B]

        # Loss is 1 - IoU so that minimising loss maximises overlap
        loss_per_sample = 1.0 - iou_per_sample

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:   # 'none'
            return loss_per_sample

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
