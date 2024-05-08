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


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, use_norm=False, stride=2):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm

        self.kernel = 4
        self.stride = stride
        self.padding = 1

        self.discriminator_block = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )

        if self.use_norm:
            layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)

        layers["lRelU"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.discriminator_block(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator Block".title())
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

    in_channels = args.in_channels
    out_channels = args.out_channels
    num_repetitive = 4

    config_files = config()

    layers = []

    for idx in tqdm(range(num_repetitive)):
        layers.append(
            DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_norm=False if idx == 0 else True,
                stride=1 if idx == num_repetitive - 1 else 2,
            )
        )
        in_channels = out_channels
        out_channels *= 2

    model = nn.Sequential(*layers)

    print(model(torch.randn((1, 3, 256, 256))).size())

    print(summary(model=model, input_size=(3, 256, 256)))

    draw_graph(
        model=model, input_data=torch.randn((1, 3, 256, 256))
    ).visual_graph.render(
        filename=os.path.join(
            config_files["path"]["files_path"], "netD_discriminator_block"
        ),
        format="jpeg",
    )
