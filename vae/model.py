import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE(nn.Module):

    def __init__(self,input_dim,hidden,output_dim):

        super().__init__()

        self.latent = nn.Linear(input_dim,hidden)
        self.latent_mean = nn.Linear(hidden,output_dim)
        self.latent_var = nn.Linear(hidden,output_dim)


        self.latent_to_h = nn.Linear(output_dim,hidden)
        self.h_to_inp = nn.Linear(hidden,input_dim)

    

    def encode(self,x):
        compressed = F.relu(self.latent(x))
        mu = self.latent_mean(compressed)
        sigma = self.latent_var(compressed)
        return mu,sigma



    def decode(self,z):

        invert = F.relu(self.latent_to_h(z))
        invert = F.sigmoid(self.h_to_inp(invert)) 
        
        return invert


    def forward(self,x):
        mu,sigma = self.encode(x)
        eps = torch.randn_like(mu)
        reparam = mu + sigma*eps
        decode = self.decode(reparam)
        return decode,mu,sigma


