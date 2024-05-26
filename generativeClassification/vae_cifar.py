import torch
import torch.nn as nn

from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.distributions import Normal, Bernoulli, Independent
from collections import defaultdict


def get_normal_KL(mean_1, log_std_1, mean_2=None, log_std_2=None):
    """
        This function should return the value of KL(p1 || p2),
        where p1 = Normal(mean_1, exp(log_std_1)), p2 = Normal(mean_2, exp(log_std_2) ** 2).
        If mean_2 and log_std_2 are None values, we will use standard normal distribution.
        Note that we consider the case of diagonal covariance matrix.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)

    std_1 = torch.exp(log_std_1)
    std_2 = torch.exp(log_std_2)

    mean_1, mean_2 = mean_1.float(), mean_2.float()
    std_1, std_2  = std_1.float(), std_2.float()

    p  = Independent(torch.distributions.Normal(mean_1, std_1), 1)
    q  = Independent(torch.distributions.Normal(mean_2, std_2), 1)
    kl = torch.distributions.kl_divergence(p, q)

    return kl

def get_normal_nll(x, mean, log_std):
    """
        This function should return the negative log likelihood log p(x),
        where p(x) = Normal(x | mean, exp(log_std) ** 2).
        Note that we consider the case of diagonal covariance matrix.
    """
    # ====
    mean = mean.float()
    std  = torch.exp(log_std).float()

    prob = Independent(torch.distributions.Normal(mean, std), reinterpreted_batch_ndims = 3)
    nnl = -prob.log_prob(x)
    return nnl

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(in_dim, 2*in_dim, kernel_size = 3, padding=1),
            nn.BatchNorm2d(2*in_dim),
            nn.ReLU(),
            nn.Conv2d(2*in_dim, out_dim, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_dim)
        )
        
        self.residual = nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding=1) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        return self.func(x) + self.residual(x)

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, n_latent, itermediate_dim = 32):
        super().__init__()
        self.input_shape = input_shape
        self.n_latent = n_latent

        self.model = nn.Sequential(
            ResBlock(input_shape[0], itermediate_dim),
            ResBlock(itermediate_dim, itermediate_dim),
            ResBlock(itermediate_dim, itermediate_dim),
            nn.MaxPool2d(2),
            ResBlock(itermediate_dim, 2*itermediate_dim),
            ResBlock(2*itermediate_dim, 2*itermediate_dim),
            nn.MaxPool2d(2),
            ResBlock(2*itermediate_dim, 4*itermediate_dim),
            ResBlock(4*itermediate_dim, 4*itermediate_dim),
            nn.MaxPool2d(2),
            ResBlock(4*itermediate_dim, 8*itermediate_dim),
            ResBlock(8*itermediate_dim, 8*itermediate_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8*itermediate_dim, 2*n_latent)
        )

    def forward(self, x):
        return self.model(x)
    
class ConvDecoder(nn.Module):
    def __init__(self, n_latent, output_shape, itermediate_dim = 32):
        super().__init__()
        self.n_latent = n_latent
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(n_latent, n_latent, stride=2, kernel_size = 2),
            ResBlock(n_latent, itermediate_dim*8),
            ResBlock(itermediate_dim*8, itermediate_dim*8),
            ResBlock(itermediate_dim*8, itermediate_dim*8),
            nn.ConvTranspose2d(itermediate_dim*8, itermediate_dim*8, stride=2, kernel_size = 2),
            ResBlock(itermediate_dim*8, itermediate_dim*4),
            ResBlock(itermediate_dim*4, itermediate_dim*4),
            ResBlock(itermediate_dim*4, itermediate_dim*4),
            nn.ConvTranspose2d(itermediate_dim*4, itermediate_dim*4, stride=2, kernel_size = 2),
            ResBlock(itermediate_dim*4, itermediate_dim*2),
            ResBlock(itermediate_dim*2, itermediate_dim*2),
            ResBlock(itermediate_dim*2, itermediate_dim*2),
            nn.ConvTranspose2d(itermediate_dim*2, itermediate_dim*2, stride=2, kernel_size = 2),
            ResBlock(itermediate_dim*2, itermediate_dim),
            ResBlock(itermediate_dim, itermediate_dim),
            ResBlock(itermediate_dim, itermediate_dim),
            nn.ConvTranspose2d(itermediate_dim, itermediate_dim, stride=2, kernel_size = 2),
            ResBlock(itermediate_dim, itermediate_dim),
            ResBlock(itermediate_dim, itermediate_dim),
            ResBlock(itermediate_dim, output_shape[0]),
        )
        

    def forward(self, z):
        b, c = z.size()
        z = z.view(b, c, 1, 1)
        res = self.model(z)
        return res
    

class ConvVAE(nn.Module):
    def __init__(self, input_shape, n_latent, beta=1, device='cuda'):
        super().__init__()
        assert len(input_shape) == 3

        self.device = device
        self.input_shape = input_shape
        self.n_latent = n_latent
        self.beta = beta
      
        self.encoder = ConvEncoder(self.input_shape, self.n_latent)
        self.decoder = ConvDecoder(self.n_latent, self.input_shape)

    def prior(self, n, use_cuda=True):
     
        z = torch.randn(n, self.n_latent).to(self.device)
        if use_cuda:
            z = z.to(self.device)
        return z

    def forward(self, x):

        mu_z, log_std_z = torch.tensor_split(self.encoder(x), 2, dim = 1)
        z = mu_z + torch.exp(log_std_z) * self.prior(x.shape[0])
        mu_x = self.decoder(z)
        return mu_z, log_std_z, mu_x
        
    def loss(self, x):

        mu_z, log_std_z, mu_x = self(x)
        recon_loss = torch.mean(get_normal_nll(x, mu_x, torch.zeros_like(mu_x)))
        kl_loss = torch.mean(get_normal_KL(mu_z, log_std_z, torch.zeros_like(mu_z), torch.zeros_like(log_std_z)))
        elbo_loss = self.beta * kl_loss + recon_loss
        dict_loss = {"recon_loss": recon_loss, "kl_loss":kl_loss, "elbo_loss":elbo_loss}
        return dict_loss

    def sample(self, n):
        with torch.no_grad():
            
            x_recon = self.decoder(self.prior(n))
            samples = torch.clamp(x_recon, -1, 1)
        return samples.cpu().numpy() * 0.5 + 0.5