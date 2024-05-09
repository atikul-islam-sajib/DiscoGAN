import sys
import os
import argparse
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(GANLoss, self).__init__()

        self.name = "GANLoss".title()
        self.reduction = reduction

        self.adversarial_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, actual, pred):
        if isinstance(actual, torch.Tensor) and isinstance(pred, torch.Tensor):
            return self.adversarial_loss(actual, pred)
        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Loss".title())
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Define the reduction type".capitalize(),
    )
    parser.add_argument(
        "--GANLoss",
        type=str,
        default="GANLoss",
        help="Define the loss type".capitalize(),
    )

    args = parser.parse_args()

    loss = GANLoss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 1.0])
    pred = torch.tensor([1.0, 0.0, 1.0])

    assert loss(actual, pred) == 0.0
