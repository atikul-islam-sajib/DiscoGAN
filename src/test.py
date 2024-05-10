import sys
import os
import argparse
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

sys.path.append("src/")

from utils import config, load, device_init
from generator import Generator


class TestModel:
    def __init__(
        self,
        dataloader="test",
        best_model=True,
        netG_XtoY=None,
        netG_YtoX=None,
        device="mps",
    ):
        self.dataloader = dataloader
        self.best_model = best_model
        self.XtoY = netG_XtoY
        self.YtoX = netG_YtoX

        self.device = device

        self.device = device_init(self.device)
        self.config = config()

        self.netG_XtoY = Generator()
        self.netG_YtoX = Generator()

        self.netG_XtoY.to(self.device)
        self.netG_YtoX.to(self.device)

    def select_best_model(self):
        if self.best_model:
            if os.path.exists(self.config["path"]["best_model_path"]):
                model_state_dict = torch.load(
                    os.path.join(
                        self.config["path"]["best_model_path"], "best_model.pth"
                    )
                )

                self.netG_XtoY.load_state_dict(model_state_dict["netG_XtoY"])
                self.netG_YtoX.load_state_dict(model_state_dict["netG_YtoX"])

            else:
                raise Exception("Cannot find the best model".capitalize())

        else:
            if isinstance(self.XtoY, Generator) and isinstance(self.YtoX, Generator):
                state_dict_XtoY = torch.load(self.XtoY)
                state_dict_YtoX = torch.load(self.YtoX)

                self.netG_XtoY.load_state_dict(state_dict_XtoY)
                self.netG_YtoX.load_state_dict(state_dict_YtoX)

            else:
                raise ValueError("XtoY and YtoX should be defined".capitalize())

    def create_gif_file(self):
        if os.path.exists(self.config["path"]["train_results"]):
            path = self.config["path"]["train_results"]

            self.images = [
                imageio.imread(os.path.join(path, image)) for image in os.listdir(path)
            ]

            if os.path.exists(self.config["path"]["gif_path"]):
                path = self.config["path"]["gif_path"]

                imageio.mimsave(
                    os.path.join(path, "train_results.gif"), self.images, "GIF"
                )

            else:
                raise Exception("Cannot create the GIF file".capitalize())

        else:
            raise Exception(
                "Cannot extract the images from the train images directory".capitalize()
            )

    def load_dataloader(self):
        if os.path.exists(self.config["path"]["processed_path"]):
            path = self.config["path"]["processed_path"]

            if self.dataloader == "all":
                self.test_dataloader = load(
                    filename=os.path.join(path, "dataloader.pkl")
                )

                return self.test_dataloader

            elif self.dataloader == "train":
                self.train_dataloader = load(
                    filename=os.path.join(path, "train_dataloader.pkl")
                )

                return self.train_dataloader

            else:
                self.dataloader = load(
                    filename=os.path.join(path, "test_dataloader.pkl")
                )

                return self.dataloader

        else:
            raise Exception("processed_path does not exist".capitalize())

    def image_normalized(self, image=None):
        if image is not None:
            return (image - image.min()) / (image.max() - image.min())

        else:
            raise ValueError("image should be a torch.Tensor".capitalize())

    def create_plot(self, **kwargs):
        plt.figure(figsize=(40, 40))

        X = kwargs["X"]
        y = kwargs["y"]

        predicted_y = self.netG_XtoY(X.to(self.device))
        reconstructed_x = self.netG_YtoX(predicted_y)

        for index, image in enumerate(predicted_y):
            real_X = X[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()
            pred_y = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            real_y = y[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()
            revert_X = reconstructed_x[index].squeeze().permute(1, 2, 0).cpu().detach()

            real_X = self.image_normalized(image=real_X)
            pred_y = self.image_normalized(image=pred_y)
            real_y = self.image_normalized(image=real_y)
            revert_X = self.image_normalized(image=revert_X)

            plt.subplot(4 * 4, 4 * 1, 4 * index + 1)
            plt.imshow(real_X)
            plt.title("X")
            plt.axis("off")

            plt.subplot(4 * 4, 4 * 1, 4 * index + 2)
            plt.imshow(pred_y)
            plt.title("pred_Y")
            plt.axis("off")

            plt.subplot(4 * 4, 4 * 1, 4 * index + 3)
            plt.imshow(real_y)
            plt.title("Y")
            plt.axis("off")

            plt.subplot(4 * 4, 4 * 1, 4 * index + 4)
            plt.imshow(revert_X)
            plt.title("Reconstructed_X")
            plt.axis("off")

        plt.tight_layout()
        if os.path.exists(self.config["path"]["test_result"]):
            path = self.config["path"]["test_result"]
            plt.savefig(os.path.join(path, "test_result.png"))
            print(
                """The result is saved as test_result.png in the "./outputs/test_result" directory"""
            )
        plt.show()

    def test(self):
        try:
            self.select_best_model()
        except Exception as e:
            print("An error occurred {}".format(e))
        else:
            self.test_dataloader = self.load_dataloader()

            X, y = next(iter(self.test_dataloader))

            self.create_plot(X=X, y=y)
            self.create_gif_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model for DiscoGAN".title())
    parser.add_argument(
        "--XtoY",
        type=str,
        default=None,
        help="Define the path to the XtoY model".capitalize(),
    )
    parser.add_argument(
        "--YtoX",
        type=str,
        default=None,
        help="Define the path to the YtoX model".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Define the device".capitalize(),
    )
    parser.add_argument(
        "--best_model",
        type=bool,
        default=True,
        help="Define whether to use the best model".capitalize(),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="test",
        help="Define the dataloader".capitalize(),
    )

    args = parser.parse_args()

    test_model = TestModel(
        dataloader=args.dataloader,
        best_model=args.best_model,
        netG_XtoY=args.XtoY,
        netG_YtoX=args.YtoX,
        device=args.device,
    )

    test_model.test()
