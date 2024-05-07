import os
import yaml
import joblib


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
