import torch
import torch.nn as nn
from tqdm import tqdm
from model import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os

input_dim = 784
hidden_dim = 200
output_dim= 32

epochs = 10
batch_size = 200
lr = 3e-4

data = datasets.FashionMNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True)

model = VAE(input_dim,hidden_dim,output_dim)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn= nn.BCELoss(reduction="sum")

def train():

    for _ in range(epochs):
        loop = tqdm(enumerate(loader))
        for i,(x,y) in loop:
            x = x.view(x.shape[0],input_dim)
            x_hat,mu,sigma = model(x)

            loss = loss_fn(x_hat,x)
            kl = -torch.sum(1+torch.log(sigma.pow(2)) -mu.pow(2)-sigma.pow(2))
            total_loss = loss+kl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loop.set_postfix(loss=total_loss.item())

    torch.save(model,'./model.pt')
    
def generate(target,num_samples):
    model = torch.load("./model.pt")
    classes = data.classes
    inp = None
    for (x,y) in data:
        if y==target:
            inp = x
            break
    mu,sigma = model.encode(inp.reshape(-1,784))

    for i in range(num_samples):
        eps = torch.randn_like(mu)
        z = mu + eps*sigma
        image = model.decode(z).view(-1,28,28)
        if target ==0:
            label = "tshirt"
        else:
            label = data.classes[target]
        save_image(image,os.path.join('./samples',f'{label}{i+1}.png'))



for i in range(10):
    generate(i,5)