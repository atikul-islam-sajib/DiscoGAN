import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import device_init, weights_init
from helper import helpers
from generator import Generator
from discriminator import Discriminator


class Trainer:
    def __init__(
        self,
        in_channels=3,
        epochs=500,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        lr_scheduler=False,
        is_display=True,
        is_weight_init=False,
    ):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.lr_scheduler = lr_scheduler
        self.is_display = is_display
        self.is_weight_init = is_weight_init

        self.init = helpers(
            lr=self.lr, adam=self.adam, SGD=self.SGD, in_channels=self.in_channels
        )

        self.device = device_init(device=self.device)

        self.netG_XtoY = self.init["netG_XtoY"]
        self.netG_YtoX = self.init["netG_YtoX"]

        self.netG_XtoY.to(self.device)
        self.netG_YtoX.to(self.device)

        self.netD_X = self.init["netD_X"]
        self.netD_Y = self.init["netD_Y"]

        self.netD_X.to(self.device)
        self.netD_Y.to(self.device)

        if self.is_weight_init:
            self.netG_XtoY.apply(weights_init)
            self.netG_YtoX.apply(weights_init)
            self.netD_X.apply(weights_init)
            self.netD_Y.apply(weights_init)

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD_X = self.init["optimizerD_X"]
        self.optimizerD_Y = self.init["optimizerD_Y"]

        self.adversarial_loss = self.init["adversarial_loss"]
        self.cycle_loss = self.init["cycle_loss"]
        self.pixel_loss = self.init["pixel_loss"]

        self.dataloader = self.init["dataloader"]
        self.train_dataloader = self.init["train_dataloader"]
        self.test_dataloader = self.init["test_dataloader"]

        print(type(self.netG_XtoY))

    def l1(self, model):
        if isinstance(model, Generator):
            return 0.01 * sum(torch.norm(params, 1) for params in model.parameters())

        else:
            raise Exception(
                "Cannot able to use L1 regularization with Generator".capitalize()
            )

    def l2(self, model):
        if isinstance(model, Generator):
            return 0.01 * sum(torch.norm(params, 2) for params in model.parameters())

        else:
            raise Exception(
                "Cannot able to use L2 regularization with Generator".capitalize()
            )

    def elastic_loss(self, model):
        if isinstance(model, Generator):
            l1 = self.l1(model=model)
            l2 = self.l2(model=model)

            return 0.01 * (l1 + l2)
        else:
            raise Exception(
                "Cannot able to use elastic regularization with Generator".capitalize()
            )

    def saved_checkpoints_netG_XtoY(self, epoch=None):
        pass

    def saved_checkpoints_netG_YtoX(self, epoch=None):
        pass

    def saved_train_best_model(self, **kwargs):
        pass

    def saved_model_history(self, **kwargs):
        pass

    def saved_train_images(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        pass

    def update_train_netG(self, **kwargs):
        pass

    def update_train_netD_X(self, **kwargs):
        pass

    def update_train_netD_Y(self, **kwargs):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer(epochs=1)
