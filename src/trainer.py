import sys
import os
import torch
import argparse
import warnings
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from helper import helpers
from generator import Generator
from discriminator import Discriminator
from utils import device_init, weights_init, config, load, dump


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
        is_save_image=True,
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
        self.is_save_image = is_save_image

        self.init = helpers(
            lr=self.lr, adam=self.adam, SGD=self.SGD, in_channels=self.in_channels
        )

        self.device = device_init(device=self.device)

        self.netG_XtoY = self.init["netG_XtoY"]
        self.netG_YtoX = self.init["netG_YtoX"]

        self.netD_X = self.init["netD_X"]
        self.netD_Y = self.init["netD_Y"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD_X = self.init["optimizerD_X"]
        self.optimizerD_Y = self.init["optimizerD_Y"]

        self.adversarial_loss = self.init["adversarial_loss"]
        self.cycle_loss = self.init["cycle_loss"]
        self.pixel_loss = self.init["pixel_loss"]

        self.dataloader = self.init["dataloader"]
        self.train_dataloader = self.init["train_dataloader"]
        self.test_dataloader = self.init["test_dataloader"]

        self.netG_XtoY.to(self.device)
        self.netG_YtoX.to(self.device)

        self.netD_X.to(self.device)
        self.netD_Y.to(self.device)

        if self.is_weight_init:
            self.netG_XtoY.apply(weights_init)
            self.netG_YtoX.apply(weights_init)
            self.netD_X.apply(weights_init)
            self.netD_Y.apply(weights_init)

        if self.lr_scheduler:
            self.schedulerG = StepLR(
                optimizer=self.optimizerG, step_size=10, gamma=0.85
            )
            self.schedulerD_X = StepLR(
                optimizer=self.optimizerD_X, step_size=10, gamma=0.85
            )
            self.schedulerD_Y = StepLR(
                optimizer=self.optimizerD_Y, step_size=10, gamma=0.85
            )

        self.total_netG_loss = []
        self.total_netD_X_loss = []
        self.total_netD_Y_loss = []

        self.history = {
            "G_loss": [],
            "D_X_loss": [],
            "D_Y_loss": [],
        }

        self.loss = float("inf")

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
        if os.path.exists(self.config["path"]["best_model_path"]):
            path = self.config["path"]["best_model_path"]

            if self.loss > np.mean(kwargs["netG_loss"]):
                self.loss = np.mean(kwargs["netG_loss"])

                torch.save(
                    {
                        "netG_XtoY": self.netG_XtoY.state_dict(),
                        "netG_YtoX": self.netG_YtoX.state_dict(),
                        "epoch": kwargs["epoch"],
                        "loss": np.mean(kwargs["netG_loss"]),
                    },
                    os.path.join(path, "best_model.pth"),
                )

        else:
            raise Exception("Cannot able to save the best_model".capitalize())

    def saved_model_history(self, **kwargs):
        if os.path.exists(self.config["path"]["files_path"]):
            path = self.config["path"]["files_path"]
            pd.DataFrame(
                {
                    "netG_loss": kwargs["netG_loss"],
                    "netD_X_loss": kwargs["netD_X_loss"],
                    "netD_Y_loss": kwargs["netD_Y_loss"],
                }
            ).to_csv(os.path.join(path, "model_history.csv"))

        else:
            raise Exception("Cannot be saved the model history".capitalize())

    def saved_train_images(self, **kwargs):
        if os.path.exists(self.config["path"]["processed_path"]):
            path = self.config["path"]["processed_path"]

            X, _ = next(iter(load(filename=os.path.join(path, "train_dataloader.pkl"))))

            predict_y = self.netG_XtoY(X.to(self.device))
            reconstructed_x = self.netG_YtoX(predict_y)

            for image in [
                ("predict_y", predict_y),
                ("reconstructed_x", reconstructed_x),
            ]:
                save_image(
                    image[1],
                    os.path.join(
                        self.config["path"]["train_results"],
                        image[0] + "{}.png".format(kwargs["epoch"]),
                    ),
                    nrow=1,
                )
        else:
            raise Exception("Cannot be saved the processed images".capitalize())

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
        else:
            print(
                "Epochs: [{}/{}] is completed".capitalize().format(
                    kwargs["epochs"], self.epochs
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
        warnings.filterwarnings("ignore")

        for epoch in tqdm(range(self.epochs)):
            netG_loss = []
            netD_X_loss = []
            netD_Y_loss = []

            for idx, (X, y) in enumerate(self.train_dataloader):
                try:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    netD_X_loss.append(self.update_train_netD_X(X=X, y=y))
                    netD_Y_loss.append(self.update_train_netD_Y(X=X, y=y))
                    netG_loss.append(self.update_train_netG(X=X, y=y))

                except Exception as e:
                    print(f"An error occurred during the training process: {e}")
                    continue

            try:
                self.show_progress(
                    netG_loss=netG_loss,
                    netD_X_loss=netD_X_loss,
                    netD_Y_loss=netD_Y_loss,
                    epoch=epoch + 1,
                )

                self.saved_checkpoints_netG_XtoY(epoch=epoch + 1)
                self.saved_checkpoints_netG_YtoX(epoch=epoch + 1)
                self.saved_train_best_model(epoch=epoch + 1, netG_loss=netG_loss)

            except Exception as e:
                print(
                    f"An error occurred while saving checkpoints or updating progress: {e}"
                )

            self.total_netG_loss.append(np.mean(netG_loss))
            self.total_netD_X_loss.append(np.mean(netD_X_loss))
            self.total_netD_Y_loss.append(np.mean(netD_Y_loss))

            if (epoch + 1) % 50 and (self.is_save_image):
                self.saved_train_images(epoch=epoch + 1)

            if self.lr_scheduler:
                self.schedulerG.step()
                self.schedulerD_X.step()
                self.schedulerD_Y.step()

        try:
            self.history["G_loss"].extend(self.total_netG_loss)
            self.history["D_X_loss"].extend(self.total_netD_X_loss)
            self.history["D_Y_loss"].extend(self.total_netD_Y_loss)

            self.saved_model_history(
                netG_loss=self.total_netG_loss,
                netD_X_loss=self.total_netD_X_loss,
                netD_Y_loss=self.total_netD_Y_loss,
            )

            if os.path.exists(self.config["path"]["metrics_path"]):
                for file in [
                    ("netG", self.total_netG_loss),
                    ("netD_X", self.total_netD_X_loss),
                    ("netD_Y", self.total_netD_Y_loss),
                ]:
                    dump(
                        value=file[1],
                        filename=os.path.join(
                            self.config["path"]["metrics_path"], file[0] + ".pkl"
                        ),
                    )

            else:
                raise Exception("Cannot be saved the metrics".capitalize())
        except Exception as e:
            print(f"An error occurred while saving the training history: {e}")

    @staticmethod
    def plot_history():
        config_files = config()
        if os.path.exists(config_files["path"]["metrics_path"]):
            path = config_files["path"]["metrics_path"]

            netG = os.path.join(path, "netG.pkl")
            netD_X = os.path.join(path, "netD_X.pkl")
            netD_Y = os.path.join(path, "netD_Y.pkl")
            files = [netG, netD_X, netD_Y]
            labels = ["Generator Loss", "Discriminator X Loss", "Discriminator Y Loss"]

            plt.figure(figsize=(20, 10))

            for index, (file, label) in enumerate(zip(files, labels)):
                if os.path.exists(file):
                    data = load(filename=file)
                    plt.subplot(1, 3, index + 1)
                    plt.plot(data, label=label)
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.title(label)
                    plt.legend()

                else:
                    print(f"Error: {file} does not exist.")

            plt.tight_layout()
            (
                plt.savefig(
                    os.path.join(
                        config_files["path"]["metrics_path"], "model_history.jpeg"
                    )
                )
                if os.path.exists(config_files["path"]["metrics_path"])
                else "Cannot be saved the image of the model history".capitalize()
            )
            plt.show()

        else:
            raise Exception("Cannot be open the metrics files".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for DiscoGAN".capitalize())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the number of channels of image".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs to train the model".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Choose the device to train the model".capitalize(),
    )
    parser.add_argument(
        "--adam", type=bool, default=True, help="Use adam optimizer".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Use SGD optimizer".capitalize()
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=False,
        help="Use the lr scheduler to prevent the exploding Gradient".capitalize(),
    )
    parser.add_argument(
        "--is_display",
        type=bool,
        default=True,
        help="Display the training process".capitalize(),
    )
    parser.add_argument(
        "--is_weight_init",
        type=bool,
        default=False,
        help="Initialize the weights of the model".capitalize(),
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )

    args = parser.parse_args()

    trainer = Trainer(
        in_channels=args.in_channels,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        adam=args.adam,
        SGD=args.SGD,
        lr_scheduler=args.lr_scheduler,
        is_display=args.is_display,
        is_weight_init=args.is_weight_init,
    )
    trainer.train()

    trainer.plot_history()
