import torch
import torch.nn as nn

from .cifar10_LeNet_vae import CIFAR10_LeNet
from base.base_net import BaseNet


class BinaryClassifierNet(BaseNet):

    def __init__(self, net_name):
        super().__init__()

        if net_name != 'cifar10_classifier':
            raise ValueError('Only cifar10_classifier is supported in this RU-VAE release.')
        self.network = CIFAR10_LeNet()

        self.linear = nn.Linear(self.network.rep_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_mu, _ = self.network(x)
        x = self.linear(x_mu)
        x = self.sigmoid(x)
        return x
