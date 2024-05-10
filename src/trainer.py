import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import device_init, weights_init, config
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

        self.total_netG_loss = []
        self.total_netD_X_loss = []
        self.total_netD_Y_loss = []

        self.config = config()

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
        if os.path.exists(self.config["path"]["netG_XtoY_path"]):
            torch.save(
                self.netG_XtoY.state_dict(),
                os.path.join(
                    self.config["path"]["netG_XtoY_path"],
                    "netG_XtoY{}.pth".format(epoch),
                ),
            )

        else:
            raise Exception("Cannot able to save the netG_XtoY model".capitalize())

    def saved_checkpoints_netG_YtoX(self, epoch=None):
        if os.path.exists(self.config["path"]["netG_YtoX_path"]):
            torch.save(
                self.netG_YtoX.state_dict(),
                os.path.join(
                    self.config["path"]["netG_YtoX_path"],
                    "netG_YtoX{}.pth".format(epoch),
                ),
            )

        else:
            raise Exception("Cannot able to save the netG_XtoY model".capitalize())

    def saved_train_best_model(self, **kwargs):
        pass

    def saved_model_history(self, **kwargs):
        pass

    def saved_train_images(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs: [{}/{}] - netG_loss: [{:.4f}] - netD_X_loss: {:.4f} - netD_Y_loss: {:.4f}".format(
                    kwargs["epoch"],
                    self.epochs,
                    np.mean(kwargs["netG_loss"]),
                    np.mean(kwargs["netD_X_loss"]),
                    np.mean(kwargs["netD_Y_loss"]),
                )
            )

    def update_train_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        fake_y = self.netG_XtoY(kwargs["X"])
        fake_y_predict = self.netD_Y(fake_y)
        fake_y_loss = self.adversarial_loss(
            fake_y_predict, torch.ones_like(fake_y_predict)
        )

        fake_x = self.netG_YtoX(kwargs["y"])
        real_x_predict = self.netD_X(fake_x)
        fake_x_loss = self.adversarial_loss(
            real_x_predict, torch.ones_like(real_x_predict)
        )

        reconstructed_x = self.netG_YtoX(fake_y)
        reconstructed_x_loss = self.cycle_loss(kwargs["X"], reconstructed_x)

        reconstructed_y = self.netG_XtoY(fake_x)
        reconstructed_y_loss = self.cycle_loss(kwargs["y"], reconstructed_y)

        pixel_loss_y = self.pixel_loss(kwargs["y"], fake_y)
        pixel_loss_x = self.pixel_loss(kwargs["X"], fake_x)

        total_G_loss = (
            (0.5 * (fake_y_loss + fake_x_loss))
            + (0.5 * (reconstructed_x_loss + reconstructed_y_loss))
            + (0.5 * (pixel_loss_x + pixel_loss_y))
        )

        total_G_loss.backward()
        self.optimizerG.step()

        return total_G_loss.item()

    def update_train_netD_X(self, **kwargs):
        self.optimizerD_X.zero_grad()

        fake_x = self.netG_YtoX(kwargs["y"])
        real_x_predict = self.netD_X(kwargs["X"])
        fake_x_predict = self.netD_X(fake_x)

        real_x_loss = self.adversarial_loss(
            real_x_predict, torch.ones_like(real_x_predict)
        )
        fake_x_loss = self.adversarial_loss(
            fake_x_predict, torch.zeros_like(fake_x_predict)
        )

        d_x_loss = (real_x_loss + fake_x_loss) / 2

        d_x_loss.backward()
        self.optimizerD_X.step()

        return d_x_loss.item()

    def update_train_netD_Y(self, **kwargs):
        self.optimizerD_Y.zero_grad()

        fake_y = self.netG_XtoY(kwargs["X"])
        real_y_predict = self.netD_Y(kwargs["y"])
        fake_y_predict = self.netD_Y(fake_y)

        real_y_loss = self.adversarial_loss(
            real_y_predict, torch.ones_like(real_y_predict)
        )
        fake_y_loss = self.adversarial_loss(
            fake_y_predict, torch.zeros_like(fake_y_predict)
        )

        d_y_loss = (real_y_loss + fake_y_loss) / 2

        d_y_loss.backward()
        self.optimizerD_Y.step()

        return d_y_loss.item()

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            netG_loss = []
            netD_X_loss = []
            netD_Y_loss = []

            for _, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                netD_X_loss.append(self.update_train_netD_X(X=X, y=y))
                netD_Y_loss.append(self.update_train_netD_Y(X=X, y=y))
                netG_loss.append(self.update_train_netG(X=X, y=y))

            self.show_progress(
                netG_loss=netG_loss,
                netD_X_loss=netD_X_loss,
                netD_Y_loss=netD_Y_loss,
                epoch=epoch + 1,
            )

            self.saved_checkpoints_netG_XtoY(epoch=epoch + 1)
            self.saved_checkpoints_netG_YtoX(epoch=epoch + 1)


if __name__ == "__main__":
    trainer = Trainer(epochs=1)
    trainer.train()
