"""
Codes for loading the MNIST data
"""
from __future__ import absolute_import, division, print_function

import os
from functools import lru_cache
from pathlib import Path

import imageio
import numpy
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

NUM_CLASSES = {
    "cifar10": 10,
    "imagenet": 1000,
}


class ImageNetDatasetValidation(torch.utils.data.Dataset):
    """This class represents the ImageNet Validation Dataset"""

    def __init__(self, trans=None, root_dir=None):

        # validation data paths
        if root_dir is None:
            self.baseDir = "/ssd7/skyeom/data/imagenet"
        else:
            self.baseDir = root_dir
        self.validationDir = os.path.join(self.baseDir, "validation")
        self.validationLabelsDir = os.path.join(self.validationDir, "info.csv")
        self.validationImagesDir = os.path.join(self.validationDir, "images")

        # read the validation labels
        self.dataInfo = pd.read_csv(self.validationLabelsDir)
        self.labels = self.dataInfo["label"].values
        self.imageNames = self.dataInfo["imageName"].values
        self.labelID = self.dataInfo["labelWNID"].values

        self.len = self.dataInfo.shape[0]

        self.transforms = trans

    # we use an lru cache in order to store the most recent
    # images that have been read, in order to minimize data access times
    @lru_cache(maxsize=128)
    def __getitem__(self, index):

        # get the filename of the image we will return
        filename = self.imageNames[index]

        # create the path to that image
        imgPath = os.path.join(self.validationImagesDir, filename)

        # load the image an an numpy array (imageio uses numpy)
        img = imageio.imread(imgPath)

        # if the image is b&w and has only one colour channel
        # create two duplicate colour channels that have the
        # same values
        if img.ndim == 2:
            img = numpy.stack([img] * 3, axis=2)

        # convert the array to a pil image, so that we can apply transformations
        img = Image.fromarray(img)

        # apply any transformations necessary
        if self.transforms is not None:
            img = self.transforms(img)

        # get the label
        labelIdx = int(self.labels[index])

        return img, labelIdx

    def __len__(self):
        return self.len


def get_mnist(datapath="../data/mnist/", download=True):
    """
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    """
    # MNIST Dataset
    train_dataset = datasets.MNIST(
        root=datapath, train=True, transform=transforms.ToTensor(), download=download
    )

    test_dataset = datasets.MNIST(
        root=datapath, train=False, transform=transforms.ToTensor()
    )
    return train_dataset, test_dataset


def get_cifar10(datapath="../../data/", download=True):
    """
    Get CIFAR10 dataset
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    # Cifar-10 Dataset
    train_dataset = datasets.CIFAR10(
        root=datapath,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                # transforms.Resize(256),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        root=datapath,
        train=False,
        transform=transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset, test_dataset


def get_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = "/ssd7/skyeom/data/imagenet"
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # we can load the training data as an ImageFolder
    train = datasets.ImageFolder(root_dir / "train", train_transform)

    # but not the validation data
    # we use the custom made ImageNetDatasetValidation class for that
    val = ImageNetDatasetValidation(val_transform, root_dir=root_dir)

    return train, val
