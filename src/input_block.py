import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import config


class InputBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(InputBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 7
        self.stride = 1
        self.padding = 3

        self.input_block = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            padding_mode="reflect",
            bias=False,
        )
        layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)
        layers["ReLU"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.input_block(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Block".title())
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
    config_files = config()

    input_block = InputBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    )

    print(input_block(torch.randn(1, 3, 256, 256)).size())

    print(summary(model=input_block, input_size=(3, 256, 256)))

    draw_graph(
        model=input_block, input_data=torch.randn(1, 3, 256, 256)
    ).visual_graph.render(
        filename=os.path.join(config_files["path"]["files_path"], "netG_input_block"),
        format="jpeg",
    )
