import sys
import os
import argparse

sys.path.append("src/")

from test import TestModel
from utils import device_init, load, dump


class Inference(TestModel):
    def __init__(self, dataloader=None, image=None, XtoY=None, YtoX=None, device="mps"):
        super(Inference, self).__init__(netG_XtoY=XtoY, netG_YtoX=YtoX, device=device)
        self.dataloader = dataloader
        self.image = image

    def single_image(self):
        pass

    def batch_images(self):
        pass


if __name__ == "__main__":
    inf = Inference()
