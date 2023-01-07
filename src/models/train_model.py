
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
from src.models.model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epochs = 2
    steps = 0
    
    model.train()
    train_losses, test_losses = [], []
    for e in range(epochs):
        print('Epoch:',e+1)
        running_loss = 0
        for i in range(500):#range(train_set['labels'].shape[1]):
            if i%100 == 0: 
                print('step:',i)
                
            images, labels = train_set['images'][i].view((1,-1)), train_set['labels'][:,i] 
            optimizer.zero_grad()          
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_losses.append(running_loss)
        print("Loss:",running_loss)
    
    torch.save(model.state_dict(), 'models/checkpoint.pth')
    
    # TODO: plotting module
    plt.plot(train_losses)
    plt.savefig(f'reports/figures/train_results_{int(time.time())}.png')
    print('Plot saved in "reports/figures/"')
    
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
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
    

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    #train()
    cli() 
