import argparse
import sys
import time

import click
import numpy as np
import torch
# extra
from matplotlib import pyplot as plt
from torch import nn

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.eval()

    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    ps = torch.exp(model(test_set['images']))
    top_p, top_class = ps.topk(1, dim=1)

    equals = top_class == test_set['labels'].view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)) 
    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(evaluate)

if __name__ == "__main__":
    cli()