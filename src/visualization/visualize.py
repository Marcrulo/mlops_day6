import argparse
import sys

import torch
from torch import nn
import click
import numpy as np

from .data import mnist
from ..models.model import MyAwesomeModel


# extra
from matplotlib import pyplot as plt
import time

@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    print("Visualize")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.eval()

    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    train_set, _ = mnist()

    ps = torch.exp(model(train_set['images']))
    plt.bar(np.arange(10),ps)
    plt.show()


cli.add_command(visualize)

if __name__ == "__main__":
    cli()