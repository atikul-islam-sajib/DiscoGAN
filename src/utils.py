import os
import yaml
import joblib
import torch
import torch.nn as nn


def config():
    with open("./config.yml", "r") as file:
        config_files = yaml.safe_load(file)

    return config_files


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cpu")


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def clear():
    config_files = config()

    netG_XtoY_path = config_files["path"]["netG_XtoY_path"]
    netG_YtoX_path = config_files["path"]["netG_YtoX_path"]
    best_model_path = config_files["path"]["best_model_path"]
    train_results = config_files["path"]["train_results"]
    test_result = config_files["path"]["test_result"]
    metrics_path = config_files["path"]["metrics_path"]

    for folder in [
        netG_XtoY_path,
        netG_YtoX_path,
        best_model_path,
        train_results,
        metrics_path,
        test_result,
    ]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
