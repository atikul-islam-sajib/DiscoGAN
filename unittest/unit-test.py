import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import config, load


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


if __name__ == "__main__":
    unittest.main()
