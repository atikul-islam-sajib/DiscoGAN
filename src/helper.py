import os
import sys
import torch
import torch.optim as optim

sys.path.append("src/")

from utils import load, config
from generator import Generator
from discriminator import Discriminator
from loss.gan_loss import GANLoss
from loss.cycle_loss import CycleLoss
from loss.pixel_loss import PixelLoss


def load_dataloader():
    if os.path.exists(config()["path"]["processed_path"]):
        path = config()["path"]["processed_path"]

        train_dataloader = load(filename=os.path.join(path, "train_dataloader.pkl"))
        test_dataloader = load(filename=os.path.join(path, "test_dataloader.pkl"))
        dataloader = load(filename=os.path.join(path, "dataloader.pkl"))

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "dataloader": dataloader,
        }

    else:
        raise Exception("Can't load dataloader".capitalize())


def helpers(**kwargs):
    in_channels = kwargs["in_channels"]
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    out_channels = 64

    netG_XtoY = Generator(in_channels=in_channels, out_channels=in_channels)
    netG_YtoX = Generator(in_channels=in_channels, out_channels=in_channels)

    netD_X = Discriminator(in_channels=in_channels, out_channels=out_channels)
    netD_Y = Discriminator(in_channels=in_channels, out_channels=out_channels)

    if adam:
        optimizerG = optim.Adam(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        optimizerD_X = optim.Adam(params=netD_X.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerD_Y = optim.Adam(params=netD_Y.parameters(), lr=lr, betas=(0.5, 0.999))

    elif SGD:
        optimizerG = optim.SGD(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
        )
        optimizerD_X = optim.SGD(params=netD_X.parameters(), lr=lr, momentum=0.95)
        optimizerD_Y = optim.SGD(params=netD_Y.parameters(), lr=lr, momentum=0.95)

    adversarial_loss = GANLoss()
    cycle_loss = CycleLoss()
    pixel_loss = PixelLoss()

    dataloader = load_dataloader()

    return {
        "netG_XtoY": netG_XtoY,
        "netG_YtoX": netG_YtoX,
        "netD_X": netD_X,
        "netD_Y": netD_Y,
        "optimizerG": optimizerG,
        "optimizerD_X": optimizerD_X,
        "optimizerD_Y": optimizerD_Y,
        "adversarial_loss": adversarial_loss,
        "cycle_loss": cycle_loss,
        "pixel_loss": pixel_loss,
        "dataloader": dataloader["dataloader"],
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
    }


if __name__ == "__main__":
    init = helpers(lr=0.0002, adam=True, SGD=False, in_channels=3)
    print(init["netG_XtoY"])
    print(init["netG_YtoX"])
    print(init["netD_X"])
    print(init["netD_Y"])
    print(init["optimizerG"])
    print(init["optimizerD_X"])
    print(init["optimizerD_Y"])
    print(init["adversarial_loss"])
    print(init["cycle_loss"])
    print(init["pixel_loss"])
    print(init["dataloader"])
    print(init["train_dataloader"])
    print(init["test_dataloader"])
