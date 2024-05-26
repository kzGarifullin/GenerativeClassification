import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
import torch.optim as optim
import torch.nn.functional as F
from generativeClassification.vqvae import Model
import argparse


DEVICE = 'cuda'
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.3*1e-3       
N_LATENS = 512      
BETA = 1
DATASET = 'CIFAR10'
PATH_FOR_PTH = 'vqvae_cifar.pth'

def test(model,test_loader):
    model.eval()
    test_loss = 0
    rec_loss = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            loss,_,loss2 = model(inputs)
            test_loss += loss.item()
            rec_loss += loss2.item()

    test_loss /= len(test_loader)
    rec_loss /= len(test_loader)
    return test_loss,rec_loss

def evaluate(model,valid_loader):
    model.eval()
    valid_loss = 0
    rec_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            loss,_,loss2 = model(inputs)
            valid_loss += loss.item()
            rec_loss += loss2.item()

    valid_loss /= len(valid_loader)
    rec_loss /= len(valid_loader)
    return valid_loss,rec_loss

def train(train_loader,valid_loader,model,optimizer, num_epochs=50, lr_scheduler=None):
    best_loss = 9999999
    for ep in range(num_epochs):
        train_loss = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            loss,_,_ = model(inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if lr_scheduler:
                lr_scheduler.step()

        train_loss /= len(train_loader)
        valid_loss,rec_loss = evaluate(model,valid_loader) 

        print(f"Epoch:{ep} |Train Loss:{train_loss}|Valid Loss:{valid_loss}|Rec Loss:{rec_loss}")

        if rec_loss <= best_loss:
            best_loss = rec_loss
            torch.save(model.state_dict(), PATH_FOR_PTH)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_dataset = None
    test_dataset = None

    if DATASET == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    accelerator = Accelerator()
    model = Model()
    optimizer = optim.Adam(model.parameters(),lr=LR)
    model,train_loader,test_loader,optimizer = accelerator.prepare(model,train_loader,test_loader,optimizer)
    train(train_loader,test_loader,model,optimizer,num_epochs=EPOCHS,lr_scheduler=None)

    torch.save(model.state_dict(), PATH_FOR_PTH)
    print("Training complete")

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=EPOCHS)
    parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=LR)
    parser.add_argument('-ld', '--latent_dim', type=int, help='Laten space dimension', default=N_LATENS)
    parser.add_argument('-ct', '--cifar_type', type=str, help='CIFAR10 or CIFAR100', default=DATASET)
    parser.add_argument('-pth', '--path', type=str, help='Weights path', default=PATH_FOR_PTH)

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    N_LATENS = args.latent_dim
    DATASET = args.cifar_type
    PATH_FOR_PTH = args.path

    main()