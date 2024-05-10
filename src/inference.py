import sys
import os
import cv2
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

sys.path.append("src/")

from test import TestModel


class Inference(TestModel):

    def __init__(
        self,
        image_size=512,
        channels=3,
        dataloader="dataloader",
        image=None,
        best_model=True,
        XtoY=None,
        YtoX=None,
        device="mps",
    ):
        super(Inference, self).__init__(
            dataloader=dataloader,
            best_model=best_model,
            netG_XtoY=XtoY,
            netG_YtoX=YtoX,
            device=device,
        )
        self.image_size = image_size
        self.channels = channels
        self.image = image

        self.batch_path = self.config["path"]["batch_results_path"]
        self.single_path = self.config["path"]["single_results_path"]

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def single_image(self):
        try:
            self.select_best_model()

            read_x = cv2.imread(self.image)
            if read_x is None:
                raise ValueError(f"Image at path {self.image} could not be loaded.")

            X = self.transforms()(Image.fromarray(read_x))
            X = X.unsqueeze(0).to(self.device)

            predict_y = self.netG_XtoY(X)
            reconstructed_x = self.netG_YtoX(predict_y)

            predict_y = predict_y.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            reconstructed_x = (
                reconstructed_x.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            )

            predict_y = self.image_normalized(image=predict_y)
            reconstructed_x = self.image_normalized(image=reconstructed_x)

            plt.imshow(predict_y)
            plt.savefig(os.path.join(self.single_path, "pred_y.png"))
            plt.close()

            plt.imshow(reconstructed_x)
            plt.savefig(os.path.join(self.single_path, "reconstructed_x.png"))
            plt.close()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def batch_images(self):
        try:
            self.select_best_model()
            self.dataloader = self.load_dataloader()

            count = 0
            for _, (X, _) in enumerate(self.dataloader):
                predicted_y = self.netG_XtoY(X.to(self.device))
                reconstructed_x = self.netG_YtoX(predicted_y)

                for idx, image in tqdm(enumerate(predicted_y)):

                    pred_y = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    pred_y = self.image_normalized(image=pred_y)

                    revert_x = (
                        reconstructed_x[idx]
                        .squeeze()
                        .permute(1, 2, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    revert_x = self.image_normalized(image=revert_x)

                    if not os.path.exists(self.batch_path):
                        os.makedirs(self.batch_path)

                    plt.imshow(pred_y)
                    plt.savefig(os.path.join(self.batch_path, f"pred_y{count + 1}.png"))
                    plt.close()

                    plt.imshow(revert_x)
                    plt.savefig(
                        os.path.join(self.batch_path, f"reconstructed_x{count + 1}.png")
                    )
                    plt.close()

                    count += 1

        except Exception as e:
            print(f"An error occurred: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for DiscoGAN".title())
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Define the path to the image".capitalize(),
    )
    parser.add_argument(
        "--best_model",
        type=bool,
        default=True,
        help="Define whether to use the best model".capitalize(),
    )
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
        "--image_size",
        type=int,
        default=512,
        help="Define the image size".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Define the number of channels of image".capitalize(),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Define the dataloader".capitalize(),
    )

    args = parser.parse_args()

    inference = Inference(
        image=args.image,
        best_model=args.best_model,
        XtoY=args.XtoY,
        YtoX=args.YtoX,
        device=args.device,
        image_size=args.image_size,
        channels=args.channels,
        dataloader=args.dataloader,
    )

    if args.image is not None:
        inference.single_image()
    else:
        inference.batch_images()
