import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from generativeClassification.vae_mnist import ConvVAE
import argparse


LATENT_DIM = 6
DEVICE = 'cuda'
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64
PATH_FOR_PTH = 'conv_vae_mnist.pth'

# Loss function for VAE
def vae_loss(reconstructed_x, x, mu, logvar):
    x = x.view(-1, 1, 28, 28)
    reconstructed_x = reconstructed_x.view(-1, 1, 28, 28)

    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train_model(model, epochs, train_loader, lr=1e-3, device='cuda'):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_arr = []

    for epoch in tqdm(range(epochs)):
        loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            # Move data to the device
            data = data.to(device)

            # Forward pass
            reconstructed_images, mu, logvar = model(data)

            # Compute loss
            loss = vae_loss(reconstructed_images, data, mu, logvar)
            loss_arr.append(loss)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(mnist_loader.dataset)} "
            #           f"({100. * batch_idx / len(mnist_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")
        print(f'Epoch {epoch}, Loss: {loss}')
    return loss_arr

def main():
    transform = transforms.ToTensor()

    # download data for training
    mnist_data = datasets.MNIST(root='./todata', train=True,
                                download=True, transform = transform)

    # set dataloader
    data_loader = torch.utils.data.DataLoader(dataset = mnist_data,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True)
    # download data for test
    test_data = datasets.MNIST(root='./data', train=False,
                                download=True, transform = transform)
    
    device = DEVICE

    latent_dim = LATENT_DIM
    conv_vae = ConvVAE(latent_dim).to(device)

    optimizer = optim.Adam(conv_vae.parameters(), lr=LR)

    mnist_data = datasets.MNIST(root='./data', train=True,
                                download=True, transform = transform)

    # set dataloader
    mnist_loader = torch.utils.data.DataLoader(dataset = mnist_data,
                                            batch_size = 64,
                                            shuffle = True)
    # download data for test
    test_data = datasets.MNIST(root='./data', train=False,
                                download=True, transform = transform)

    # Training loop
    epochs = EPOCHS
    loss_arr = train_model(conv_vae, epochs, mnist_loader, device=device)

    torch.save(conv_vae.state_dict(), PATH_FOR_PTH)

    print("Training complete")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, help='Device for training', default=DEVICE)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=EPOCHS)
    parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=LR)
    parser.add_argument('-ld', '--latent_dim', type=int, help='Laten space dimension', default=LATENT_DIM)
    parser.add_argument('-pth', '--path', type=str, help='Weights path', default=PATH_FOR_PTH)

    args = parser.parse_args()

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LATENT_DIM = args.latent_dim
    PATH_FOR_PTH = args.path

    main()