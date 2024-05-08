import sys
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import config
from input_block import InputBlock
from encoder import EncoderBlock
from decoder import DecoderBlock
from residual_block import ResidualBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64

        self.kernel = 7
        self.stride = 1
        self.padding = 3

        self.layers = []

        self.layers.append(
            InputBlock(in_channels=self.in_channels, out_channels=self.out_channels)
        )
        self.in_channels = self.out_channels
        self.out_channels *= 2

        for _ in tqdm(range(2)):
            self.layers.append(
                EncoderBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )

            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.in_channels = self.in_channels
        self.out_channels //= 2

        for _ in tqdm(range(9)):
            self.layers.append(
                ResidualBlock(
                    in_channels=self.in_channels, out_channels=self.in_channels
                )
            )

        self.out_channels //= 2

        for _ in tqdm(range(2)):
            self.layers.append(
                DecoderBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )

            self.in_channels = self.out_channels
            self.out_channels //= 2

        self.model = nn.Sequential(*self.layers)

        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channels * 2,
                out_channels=self.in_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.model(x)
            return self.output(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Generator for the DiscoGAN".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the number of input channels".capitalize(),
    )
    args = parser.parse_args()
    config_files = config()

    in_channels = args.in_channels

    netG = Generator(in_channels=in_channels)

    print(netG(torch.randn((1, 3, 256, 256))).size())

    print(summary(model=netG, input_size=(3, 256, 256)))

    draw_graph(
        model=netG, input_data=torch.randn((1, 3, 256, 256))
    ).visual_graph.render(
        filename=os.path.join(config_files["path"]["files_path"], "netG1_model"),
        format="jpeg",
    )
