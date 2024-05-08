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
from discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = 4
        self.stride = 1
        self.padding = 1

        self.layers = []

        for idx in tqdm(range(4)):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    use_norm=False if idx == 0 else True,
                    stride=1 if idx == 3 else 2,
                )
            )
            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.model = nn.Sequential(*self.layers)

        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=1,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.model(x)
            return self.output(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Define the number of output channels".capitalize(),
    )
    args = parser.parse_args()

    config_files = config()

    in_channels = args.in_channels
    out_channels = args.out_channels

    netD = Discriminator(in_channels=in_channels, out_channels=out_channels)

    print(netD(torch.randn(1, 3, 256, 256)).size())
    print(summary(model=netD, input_size=(3, 256, 256)))
    draw_graph(
        model=netD, input_data=torch.randn((1, 3, 256, 256))
    ).visual_graph.render(
        filename=os.path.join(config_files["path"]["files_path"], "netD_model"),
        format="jpeg",
    )
