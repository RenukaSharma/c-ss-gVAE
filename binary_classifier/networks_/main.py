from .cifar10_LeNet_vae import CIFAR10_LeNet_Autoencoder
from .malaria_net import Malaria_net_Ext
from .nanofibre_LeNet_vae import Nanofibre_LeNet
from .binary_classifier_net import BinaryClassifierNet


def build_autoencoder(net_name):
    """Builds the network used by the binary-classifier baseline."""

    implemented_networks = ('cifar10_LeNet', 'malaria_net', 'nanofibre_vae', 'cifar10_classifier')
    assert net_name in implemented_networks

    if 'classifier' in net_name:
        return BinaryClassifierNet(net_name)

    if net_name == 'cifar10_LeNet':
        return CIFAR10_LeNet_Autoencoder()

    if net_name == 'malaria_net':
        return Malaria_net_Ext()

    if net_name == 'nanofibre_vae':
        return Nanofibre_LeNet()

    return None
