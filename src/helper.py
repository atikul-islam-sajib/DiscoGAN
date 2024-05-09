import os
import sys

sys.path.append("src/")

from utils import load
from loss.gan_loss import GANLoss
from loss.cycle_loss import CycleLoss
from loss.pixel_loss import PixelLoss


def load_dataloader():
    pass


def helpers(**kwargs):
    in_channels = kwargs["in_channels"]
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    # For Adam

    # For SGD

    # Loss

    # dataloader

    # Return


if __name__ == "__main__":
    helpers(lr=0.0002, adam=True, SGD=False, in_channels=3)
