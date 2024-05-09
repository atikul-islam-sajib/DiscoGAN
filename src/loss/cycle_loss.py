import sys
import os
import argparse
import torch
import torch.nn as nn


class CycleLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CycleLoss, self).__init__()

        self.name = "CycleLoss".title()
        self.reduction = reduction

        self.cycle_loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, actual, pred):
        if isinstance(actual, torch.Tensor) and isinstance(pred, torch.Tensor):
            return self.cycle_loss(actual, pred)
        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cycle Loss".title())
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Define the reduction type".capitalize(),
    )
    parser.add_argument(
        "--CycleLoss",
        type=str,
        default="CycleLoss",
        help="Define the loss type".capitalize(),
    )
    args = parser.parse_args()

    loss = CycleLoss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    pred = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    assert loss(actual, pred) == 0.0
