import sys
import yaml
import argparse

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import TestModel
from inference import Inference


def cli():
    parser = argparse.ArgumentParser(description="CLI for the DiscoGAN".title())
    parser.add_argument(
        "--image_path", type=str, default=None, help="Define the data path".capitalize()
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--image_size", type=int, default=512, help="Define the Image size".capitalize()
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
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Define the path to the image".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )
    parser.add_argument(
        "--single_image",
        action="store_true",
        help="Inference the single image with model".capitalize(),
    )
    parser.add_argument(
        "--batch_image",
        action="store_true",
        help="Inference the batch image with model".capitalize(),
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

    if args.train:

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
        loader.extract_features()
        loader.create_dataloader()

        loader.dataset_details()
        loader.plot_images()

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

        with open("./trained_params.yml", "w") as file:
            yaml.safe_dump(
                file,
                {
                    "train_params": {
                        "image_path": args.image_path,
                        "batch_size": args.batch_size,
                        "image_size": args.image_size,
                        "split_size": args.split_size,
                        "channels": args.channels,
                        "unpaired_images": args.unpaired_images,
                        "paired_images": args.paired_images,
                        "in_channels": args.in_channels,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "device": args.device,
                        "adam": args.adam,
                        "SGD": args.SGD,
                        "lr_scheduler": args.lr_scheduler,
                        "is_display": args.is_display,
                        "is_weight_init": args.is_weight_init,
                    }
                },
            )

    elif args.test:
        test_model = TestModel(
            dataloader=args.dataloader,
            best_model=args.best_model,
            netG_XtoY=args.XtoY,
            netG_YtoX=args.YtoX,
            device=args.device,
        )

        test_model.test()

    elif args.single_image:
        if args.image is not None:
            inference.single_image()

    elif args.batch_image:
        if args.image is None:
            inference.batch_images()


if __name__ == "__main__":
    cli()
