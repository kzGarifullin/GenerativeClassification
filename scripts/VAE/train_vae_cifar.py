from collections import defaultdict
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms, datasets
from generativeClassification.vae_cifar import ConvVAE
import argparse

DEVICE = 'cuda'
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.3*1e-3       
N_LATENS = 512      
BETA = 1
DATASET='CIFAR10'
PATH_FOR_PTH = 'conv_vae_cifar.pth'

def train_epoch(model, train_loader, optimizer, use_cuda, loss_key='total'):
    model.train()

    stats = defaultdict(list)
    for batch_idx, (x, _) in enumerate(train_loader):
        if use_cuda:
            x = x.to(DEVICE)
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model, train_loader, use_cuda):
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(train_loader):
            if use_cuda:
                x = x.to(DEVICE)
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(train_loader.dataset)
    return stats


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    use_tqdm=False,
    use_cuda=False,
    loss_key='total_loss', 
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.to(DEVICE)

    k = 0
    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model, test_loader, use_cuda)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
        print(f"{test_losses['elbo_loss']=}")
        print(f"{test_losses['kl_loss']=}")
        print(f"{test_losses['recon_loss']=}")

    return dict(train_losses), dict(test_losses)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images to [-1, 1]
    ])
 
    train_dataset = None
    test_dataset = None

    if DATASET == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = ConvVAE((3, 32, 32), N_LATENS, BETA, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, test_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        epochs=EPOCHS, 
        lr=LR, 
        loss_key='elbo_loss', 
        use_tqdm=True, 
        use_cuda=True, 
    )

    torch.save(model.state_dict(), PATH_FOR_PTH)

    print("Training complete")


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, help='Device for training', default=DEVICE)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=EPOCHS)
    parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=LR)
    parser.add_argument('-ld', '--latent_dim', type=int, help='Laten space dimension', default=N_LATENS)
    parser.add_argument('-ct', '--cifar_type', type=str, help='CIFAR10 or CIFAR100', default=DATASET)
    parser.add_argument('-pth', '--path', type=str, help='Weights path', default=PATH_FOR_PTH)


    args = parser.parse_args()

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    N_LATENS = args.latent_dim
    DATASET= args.cifar_type
    PATH_FOR_PTH = args.path

    main()