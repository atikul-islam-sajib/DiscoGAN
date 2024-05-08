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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 3
        self.stride = 2
        self.padding = 1

        self.encoder_block = self.block()

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
        layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)
        layers["ReLU"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder_block(x)

        else:
            raise Exception("Unable to process the input".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoder Block".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Define the number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=128,
        help="Define the number of output channels".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels
    num_repetitive = 2

    config_files = config()

    layers = []

    for _ in tqdm(range(num_repetitive)):
        layers.append(EncoderBlock(in_channels=in_channels, out_channels=out_channels))

        in_channels = out_channels
        out_channels *= 2

    model = nn.Sequential(*layers)

    print(model(torch.randn(1, 64, 256, 256)).size())
    print(summary(model=model, input_size=(64, 256, 256)))
    draw_graph(
        model=model, input_data=torch.randn(1, 64, 256, 256)
    ).visual_graph.render(
        filename=os.path.join(config_files["path"]["files_path"], "netG_encoder_block"),
        format="jpeg",
    )
