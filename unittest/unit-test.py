import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import config, load
from generator import Generator
from discriminator import Discriminator
from helper import helpers


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.train_dataloader = load(
            os.path.join(config()["path"]["processed_path"], "train_dataloader.pkl")
        )
        self.test_dataloader = load(
            os.path.join(config()["path"]["processed_path"], "test_dataloader.pkl")
        )
        self.dataloader = load(
            os.path.join(config()["path"]["processed_path"], "dataloader.pkl")
        )
        self.netG = Generator(in_channels=3)
        self.netD = Discriminator(in_channels=3, out_channels=64)
        self.helper = helpers(lr=0.0002, adam=True, SGD=False, in_channels=3)

    def test_train_dataloader_shape(self):
        data, mask = next(iter(self.train_dataloader))

        self.assertEqual(data.size(), torch.Size([1, 3, 256, 256]))
        self.assertEqual(mask.size(), torch.Size([1, 3, 256, 256]))

    def test_test_dataloader_shape(self):
        data, mask = next(iter(self.test_dataloader))

        self.assertEqual(data.size(), torch.Size([4, 3, 256, 256]))
        self.assertEqual(mask.size(), torch.Size([4, 3, 256, 256]))

    def test_dataloader_shape(self):
        data, mask = next(iter(self.dataloader))

        self.assertEqual(data.size(), torch.Size([8, 3, 256, 256]))
        self.assertEqual(mask.size(), torch.Size([8, 3, 256, 256]))

    def total_data_quantity(self):
        self.assertEqual(
            sum(data.size(0) for data, _ in self.dataloader),
            sum(data.size(0) for data, _ in self.train_dataloader)
            + sum(data.size(0) for data, _ in self.test_dataloader),
        )

    def test_train_quantity(self):
        self.assertEqual(
            25,
            sum(data.size(0) for data, _ in self.train_dataloader),
        )

    def test_test_dataloader_shape(self):
        self.assertEqual(
            7,
            sum(data.size(0) for data, _ in self.test_dataloader),
        )

    def test_netG_shape(self):
        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 3, 256, 256])
        )

    def test_netD_shape(self):
        self.assertEqual(
            self.netD(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 1, 30, 30])
        )

    def test_total_params_netG(self):
        self.assertEqual(self.netG.total_params(model=self.netG), 11372928)

    def test_helpers_netG_XtoY(self):
        self.netG_XtoY = self.helper["netG_XtoY"]

        self.assertEqual(
            self.netG_XtoY(torch.randn(1, 3, 256, 256)).size(),
            torch.Size([1, 3, 256, 256]),
        )

    def test_helpers_netG_YtoX(self):
        self.netG_YtoX = self.helper["netG_YtoX"]

        self.assertEqual(
            self.netG_YtoX(torch.randn(1, 3, 256, 256)).size(),
            torch.Size([1, 3, 256, 256]),
        )

    def test_helpers_netD_X_shape(self):
        self.netD_X = self.helper["netD_X"]

        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 3, 256, 256])
        )

    def test_helpers_netD_Y_shape(self):
        self.netD_Y = self.helper["netD_Y"]

        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 3, 256, 256])
        )


if __name__ == "__main__":
    unittest.main()
