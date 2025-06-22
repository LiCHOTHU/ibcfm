#!/usr/bin/env python3
"""
download_only.py

Downloads CIFAR-10, MNIST, and Fashion-MNIST datasets into ./data.
"""
from torchvision import datasets

def download_datasets(root: str = "./data"):
    # CIFAR-10
    datasets.CIFAR10(root=f"{root}/cifar10", train=True,  download=True)
    datasets.CIFAR10(root=f"{root}/cifar10", train=False, download=True)
    print("→ CIFAR-10 downloaded to", f"{root}/cifar10")

    # MNIST
    datasets.MNIST     (root=f"{root}/mnist",       train=True,  download=True)
    datasets.MNIST     (root=f"{root}/mnist",       train=False, download=True)
    print("→ MNIST downloaded to", f"{root}/mnist")

    # Fashion-MNIST
    datasets.FashionMNIST(root=f"{root}/fashion-mnist", train=True,  download=True)
    datasets.FashionMNIST(root=f"{root}/fashion-mnist", train=False, download=True)
    print("→ Fashion-MNIST downloaded to", f"{root}/fashion-mnist")

if __name__ == "__main__":
    download_datasets()
