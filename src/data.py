"""
Codes for loading the MNIST data
"""
from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import datasetops as do
import imageio
import numpy
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import datasets, transforms

NUM_CLASSES = {
    "catsanddogs": 2,
    "cifar10": 10,
    "cifar100": 100,
    "oxfordflowers102": 102,
    "stanfordcars": 196,
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


def get_cifar10(datapath="./datasets/", download=True, image_size=32):
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
                transforms.RandomResizedCrop(image_size),
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
                transforms.Resize(int(256 / 224 * image_size)),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset, test_dataset


def get_cifar100(datapath="./datasets/", download=True, image_size=32):
    """
    Get CIFAR100 dataset
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    # Cifar-100 Dataset
    train_dataset = datasets.CIFAR100(
        root=datapath,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=download,
    )

    test_dataset = datasets.CIFAR100(
        root=datapath,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(int(256 / 224 * image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset, test_dataset


def get_stanfordcars(datapath="./datasets/", download=False, image_size=224):
    """
    Get StanfordCars dataset

    NB: Automatic download fails. The dataset can be manually downloaded at
    https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

    The test labels should be places in the root folder and can retrieved at
    https://github.com/nguyentruonglau/stanford-cars/blob/main/labeldata/cars_test_annos_withlabels.mat

    To comply with the expected structure in torchvision, you also need to
    1) move the "~/cars_devkit/devkit" folder to "~/devkit".
    2) move the "~/cars_devkit/cars_train" folder to "~/cars_train".
    3) move the "~/cars_devkit/cars_test" folder to "~/cars_test".
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    train_dataset = datasets.StanfordCars(
        root=datapath,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=download,
    )

    test_dataset = datasets.StanfordCars(
        root=datapath,
        split="test",
        transform=transforms.Compose(
            [
                transforms.Resize(int(256 / 224 * image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset, test_dataset


def get_catsanddogs(datapath="./datasets/", download=True, image_size=224):
    dataset_path = Path(datapath) / "catsanddogs"

    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

    if download:
        if not dataset_path.exists():
            datasets.utils.download_and_extract_archive(
                dataset_url, dataset_path, dataset_path
            )
            (dataset_path / "PetImages/Dog/Thumbs.db").unlink(missing_ok=True)
            (dataset_path / "PetImages/Cat/Thumbs.db").unlink(missing_ok=True)
            # Images not loadable with PIL:
            (dataset_path / "PetImages/Dog/11702.jpg").unlink(missing_ok=True)
            (dataset_path / "PetImages/Cat/666.jpg").unlink(missing_ok=True)

    train, test = (
        do.from_folder_class_data(dataset_path / "PetImages")
        .named("data", "label")
        .categorical("label")
        .image("data")
        .shuffle(seed=42)
        .split([0.7, 0.3])
    )

    # Restrict the datasets length to match the KMLC-Challenge-1
    # As described in "Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning" suplemental material
    train._ids = train._ids[: 4000 + 4005]
    test._ids = test._ids[:2023]

    # Apply PyTorch relevant transforms
    def fix_shape(x):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if x.shape[0] == 4:
            x = x[:3]
        assert x.shape[0] == 3
        assert len(x.shape) == 3
        return x

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fix_shape,
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(int(256 / 224 * image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            fix_shape,
            normalize,
        ]
    )

    train_ds = train.transform(lambda x: (train_transform(x[0]), x[1])).to_pytorch()
    test_ds = test.transform(lambda x: (test_transform(x[0]), x[1])).to_pytorch()

    return train_ds, test_ds


def get_oxfordflowers102(datapath="./datasets/", download=True, image_size=224):
    dataset_path = Path(datapath) / "oxfordflowers102"

    images_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    splits_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    if download:
        if not dataset_path.exists():
            datasets.utils.download_and_extract_archive(
                images_url, dataset_path, dataset_path / "102flowers"
            )
            datasets.utils.download_url(labels_url, dataset_path, "imagelabels.mat")
            datasets.utils.download_url(splits_url, dataset_path, "setid.mat")

    # Parse labels
    labels = list(loadmat(dataset_path / "imagelabels.mat")["labels"].squeeze(0))
    splits = loadmat(dataset_path / "setid.mat")
    # In "Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning" suplemental material
    # they describe over 2000 training images. I assume, they used the val set for training here as well.
    train_ids = list(splits["trnid"].squeeze(0)) + list(splits["valid"].squeeze(0))
    assert len(train_ids) == 2040
    test_ids = list(splits["tstid"].squeeze(0))
    assert len(test_ids) == 6149

    # Ensure enumeration starts from 0
    train_ids = [i - 1 for i in train_ids]
    test_ids = [i - 1 for i in test_ids]

    # Create loaders
    image_loader = do.from_folder_data(dataset_path / "102flowers" / "jpg")
    image_loader._ids = sorted(image_loader._ids)
    label_loader = do.Loader(lambda x: (x,))
    label_loader.extend(labels)
    ds = do.zipped(image_loader, label_loader).named("data", "labels")

    # Create splits
    train = deepcopy(ds).categorical("labels").image("data")
    test = deepcopy(ds).categorical("labels").image("data")
    train._ids = train_ids
    test._ids = test_ids

    # Apply PyTorch relevant transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(int(256 / 224 * image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = train.transform(lambda x: (train_transform(x[0]), x[1])).to_pytorch()
    test_ds = test.transform(lambda x: (test_transform(x[0]), x[1])).to_pytorch()

    return train_ds, test_ds


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


if __name__ == "__main__":
    get_catsanddogs()
