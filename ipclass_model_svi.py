


''' Probabilistic Classifier '''

# imports

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch import autograd, nn, tanh, optim
import torch.nn.functional as F
from torchvision import transforms
import h5py
import numpy as np 
from pathlib import Path

import pyro
from pyro.distributions import Normal, Categorical, Laplace
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# parameters

batch_size = 2
gmu_dim = 2
out_dim = 2
learning_rate = 0.1
num_epochs = 200

# model

class IPClass(nn.Module):
    # GMU
    def __init__(self, gmu_dim, out_dim):
        super(IPClass, self).__init__()
        self.fc0 = nn.Linear(gmu_dim, out_dim)
        self.fc1 = nn.Linear(out_dim, out_dim)

    def forward(self, x_gmu, x_label):
        x = self.fc0(x_gmu.float().view(-1, 2))
        x = torch.tanh(x)
        x = self.fc1(x, x_label)
        x = torch.tanh(x)
        out = torch.tanh(x)
        return out

    # pyro model
    def model(x_gmu, x_label):
        def model_dist(*shape):
            loc = torch.zeros(*shape)
            scale = torch.ones(*shape)
            return Laplace(loc, scale)
        priors = {
            'fc0.weight': model_dist(gmu_dim, out_dim), 'fc0.bias': model_dist(gmu_dim),
            'fc1.weight': model_dist(out_dim, out_dim), 'fc1.bias': model_dist(out_dim)}
        lifted_module = pyro.random_module("net", net, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate("map"):
            x_gmu = x_gmu
            x_label = x_label
        lhat = torch.sigmoid(lifted_reg_model(x_gmu, x_label))
        pyro.sample("obs", Categorical(logits=lhat), obs=x_label)

    # pyro guide
    def guide(x_gmu, x_label):
        def infer_dist(name, *shape):
            l = torch.empty(*shape, requires_grad=True)
            s = torch.empty(*shape, requires_grad=True)
            torch.nn.init.normal_(l, std=0.01)
            torch.nn.init.normal_(s, std=0.01)
            loc = pyro.param(name+"_loc", l)
            scale = nn.functional.softplus(pyro.param(name+"_scale", s))
            return Laplace(loc, scale)
        dists = {
            'fc0.weight': infer_dist("W1", gmu_dim, out_dim), 'fc0.bias': infer_dist("b1", out_dim),
            'fc1.weight': infer_dist("W2", out_dim, out_dim), 'fc1.bias': infer_dist("b2", out_dim)}
        lifted_module = pyro.random_module("net", net, dists)
        with pyro.plate("map"):
            x_gmu = x_gmu
            x_label = x_label
        return lifted_module()

# clear param store
pyro.clear_param_store()

# instantiate the model and optimiser
net = IPClass(gmu_dim=gmu_dim, out_dim=out_dim)

# pyro svi, initialise losses

inference = SVI(IPClass.model, IPClass.guide, Adam({"lr": learning_rate}), loss=Trace_ELBO())
losses = []

# training loop

for epoch in range(num_epochs):
    loss = 0.0
    # calculate outputs
    for batch, (batch_gmu, batch_label) in enumerate(zip(train_loader_gmu, train_loader_label)):
        x_gmu = batch_t['x_gmu']
        x_label = batch_tar['x_label']
        loss = inference.step(x_gmu, x_label) # svi step
        losses.append(loss)
    # epoch loss
    normalizer_train = len(dataset_t)
    total_epoch_loss_train = loss / normalizer_train
    print('total_epoch_loss_train', total_epoch_loss_train)