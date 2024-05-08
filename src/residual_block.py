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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 3
        self.stride = 1
        self.padding = 1

        self.residual_block = self.block()

    def block(self):
        layers = OrderedDict()
        for idx in range(2):
            layers["conv{}".format(idx + 1)] = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                padding=self.padding,
                bias=False,
            )
            layers["instance_norm{}".format(idx + 1)] = nn.InstanceNorm2d(
                num_features=self.out_channels
            )

            if idx == 0:
                layers["ReLU"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return x + self.residual_block(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Block".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=256,
        help="Define the number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=256,
        help="Define the number of output channels".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    num_repetitive = args.out_channels

    config_files = config()
    layers = []

    for idx in tqdm(range(num_repetitive)):
        layers += [ResidualBlock(in_channels=in_channels, out_channels=in_channels)]

    model = nn.Sequential(*layers)
    print(model(torch.randn(1, 256, 64, 64)).size())
    print(summary(model=model, input_size=(256, 64, 64)))
    draw_graph(model=model, input_data=torch.randn(1, 256, 64, 64)).visual_graph.render(
        filename=os.path.join(
            config_files["path"]["files_path"], "netG_residual_block"
        ),
        format="jpeg",
    )
