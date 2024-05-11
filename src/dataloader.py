import sys
import argparse
import os
import zipfile
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import config, dump, load


class Loader:
    def __init__(
        self,
        image_path=None,
        image_size=256,
        channels=3,
        batch_size=1,
        split_size=0.20,
        paired_images=False,
        unpaired_images=True,
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.split_size = split_size
        self.paired_image = paired_images
        self.unpaired_image = unpaired_images

        self.config = config()

        self.X = []
        self.y = []

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Grayscale(num_output_channels=self.channels),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def data_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def unzip_folder(self):
        if os.path.exists(self.config["path"]["raw_path"]):
            path = self.config["path"]["raw_path"]

            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(
                    os.path.join(
                        path,
                    )
                )

        else:
            raise Exception(
                "Unable to unzip the folder as the path is not exists".capitalize()
            )

    def extract_features(self):
        self.directory = os.path.join(self.config["path"]["raw_path"])
        self.categories = os.listdir(os.path.join(self.directory, "images"))

        for index, category in (
            enumerate(self.categories[0])
            if self.paired_image
            else enumerate(self.categories)
        ):
            path = os.path.join(self.directory, "images", category)

            for image in os.listdir(path):

                if self.paired_image:
                    if image in os.listdir(os.path.join(self.directory, "images", "y")):
                        image_path_X = os.path.join(path, image)
                        image_path_y = os.path.join(
                            self.directory, "images", "y", image
                        )

                        image_X = cv2.imread(image_path_X)
                        image_y = cv2.imread(image_path_y)

                        image_X = cv2.cvtColor(image_X, cv2.COLOR_BGR2RGB)
                        image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)

                        self.X.append(self.transforms()(Image.fromarray(image_X)))
                        self.y.append(self.transforms()(Image.fromarray(image_y)))

                elif self.unpaired_image:
                    image_path = os.path.join(path, image)

                    image_read = cv2.imread(image_path)
                    image_read = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)

                    (
                        self.X.append(self.transforms()(Image.fromarray(image_read)))
                        if index % 2
                        else self.y.append(
                            self.transforms()(Image.fromarray(image_read))
                        )
                    )

        data = self.data_split(self.X, self.y)

        return data

    def create_dataloader(self):

        self.data = self.extract_features()

        if os.path.exists(self.config["path"]["processed_path"]):
            processed_path = self.config["path"]["processed_path"]

            train_dataloader = DataLoader(
                dataset=list(zip(self.data["X_train"], self.data["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            test_dataloader = DataLoader(
                dataset=list(zip(self.data["X_test"], self.data["y_test"])),
                batch_size=self.batch_size * 4,
                shuffle=True,
            )

            dataloader = DataLoader(
                dataset=list(zip(self.X, self.y)),
                batch_size=self.batch_size * 8,
                shuffle=True,
            )

            dump(
                value=train_dataloader,
                filename=os.path.join(processed_path, "train_dataloader.pkl"),
            )

            dump(
                value=test_dataloader,
                filename=os.path.join(processed_path, "test_dataloader.pkl"),
            )

            dump(
                value=dataloader,
                filename=os.path.join(processed_path, "dataloader.pkl"),
            )

        else:
            raise Exception("Unable to create the pickle file".capitalize())

    @staticmethod
    def plot_images():
        config_files = config()

        if os.path.exists(config_files["path"]["processed_path"]):
            dataloader = load(
                os.path.join(config_files["path"]["processed_path"], "dataloader.pkl")
            )

            X, y = next(iter(dataloader))

            plt.figure(figsize=(10, 10))

            for index, image in enumerate(X):
                image_X = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                image_y = y[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()

                image_X = (image_X - image_X.min()) / (image_X.max() - image_X.min())
                image_y = (image_y - image_y.min()) / (image_y.max() - image_y.min())

                plt.subplot(2 * 4, 2 * 2, 2 * index + 1)
                plt.imshow(image_X)
                plt.axis("off")

                plt.subplot(2 * 4, 2 * 2, 2 * index + 2)
                plt.imshow(image_y)
                plt.axis("off")

            plt.tight_layout()

            if os.path.exists(config_files["path"]["files_path"]):
                plt.savefig(
                    os.path.join(config_files["path"]["files_path"], "images.png")
                )
            else:
                raise Exception(
                    "Unable to save the images as the path is not exists".capitalize()
                )

            plt.show()

        else:
            raise Exception(
                "Unable to plot the images as the path is not exists".capitalize()
            )

    @staticmethod
    def dataset_details():
        config_files = config()

        if os.path.exists(config_files["path"]["processed_path"]):
            path = config_files["path"]["processed_path"]

            train_dataloader = load(filename=os.path.join(path, "train_dataloader.pkl"))
            test_dataloader = load(filename=os.path.join(path, "test_dataloader.pkl"))
            dataloader = load(filename=os.path.join(path, "dataloader.pkl"))

            pd.DataFrame(
                {
                    "train_data(total)": str(
                        sum(data.size(0) for data, _ in train_dataloader)
                    ),
                    "test_data(total)": str(
                        sum(data.size(0) for data, _ in test_dataloader)
                    ),
                    "data(total)": str(sum(data.size(0) for data, _ in dataloader)),
                    "train_data(batch)": str(len(train_dataloader)),
                    "test_data(batch)": str(len(test_dataloader)),
                    "train_data(shape)": str(train_dataloader.dataset[0][0].shape),
                    "test_data(shape)": str(test_dataloader.dataset[0][0].shape),
                    "data(shape)": str(dataloader.dataset[0][0].shape),
                },
                index=["Details dataset".capitalize()],
            ).T.to_csv(
                os.path.join(
                    os.path.join(
                        config_files["path"]["files_path"], "dataset_details.csv"
                    )
                    if os.path.exists(config_files["path"]["files_path"])
                    else os.path.join(
                        config_files["path"]["files_path"], "dataset_details.csv"
                    )
                )
            )

        else:
            raise Exception("Unable to create the pickle file".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader for the DiscoGAN".title())
    parser.add_argument(
        "--image_path", type=str, default=None, help="Define the data path".capitalize()
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Define the Image size".capitalize()
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.20,
        help="Define the split size".capitalize(),
    )
    parser.add_argument(
        "--channels", type=int, default=3, help="Define the channels".capitalize()
    )
    parser.add_argument(
        "--unpaired_images",
        type=bool,
        default=False,
        help="Define whether the image is unpaired or not".capitalize(),
    )
    parser.add_argument(
        "--paired_images",
        type=bool,
        default=True,
        help="Define whether the image is unpaired or not".capitalize(),
    )
    args = parser.parse_args()

    if args.image_path:
        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            split_size=args.split_size,
            channels=args.channels,
            unpaired_images=args.unpaired_images,
            paired_images=args.paired_images,
        )

        loader.unzip_folder()
        # loader.extract_features() # No need to call as it is calling from the "create_dataloader" method
        loader.create_dataloader()

        loader.dataset_details()
        loader.plot_images()
    else:
        raise Exception("Unable to create the dataloader".capitalize())
