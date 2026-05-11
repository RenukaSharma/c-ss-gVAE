from .cifar10_LeNet_vae import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_Decoder
from .nanofibre_vae import (
    Nanofibre_LeNet,
    Nanofibre_LeNet_Autoencoder,
    Nanofibre_LeNet_Decoder,
)
from .malaria_net import Malaria_net_Ext


def build_network(net_name, ae_net=None):
    """Builds the neural network phi (and optional separate decoder for hybrid training)."""

    implemented_networks = ('cifar10_LeNet', 'malaria_net', 'nanofibre_vae')
    assert net_name in implemented_networks

    net = None
    dec_net = None

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()
        dec_net = CIFAR10_LeNet_Decoder()

    if net_name == 'malaria_net':
        net = Malaria_net_Ext()

    if net_name == 'nanofibre_vae':
        net = Nanofibre_LeNet()
        dec_net = Nanofibre_LeNet_Decoder()

    return net, dec_net


def build_autoencoder(net_name):
    """Builds the autoencoder (or unified encoder–decoder) used for pretraining."""

    implemented_networks = ('cifar10_LeNet', 'malaria_net', 'nanofibre_vae')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'malaria_net':
        ae_net = Malaria_net_Ext()

    if net_name == 'nanofibre_vae':
        ae_net = Nanofibre_LeNet_Autoencoder()

    return ae_net
