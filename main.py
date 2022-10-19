# -*- encoding: utf-8 -*-
"""
@reference: Seul-Ki Yeom et al., "Pruning by explaining: a novel criterion for deep neural network pruning," Pattern Recognition, 2020.
@author: Seul-Ki Yeom, Philipp Seegerer, Sebastian Lapuschkin, Alexander Binder, Simon Wiedemann, Klaus-Robert Müller, Wojciech Samek
"""

from __future__ import print_function

import argparse

import torch
from time import sleep
import lrp_pruning.prune_resnet as modules_resnet
import lrp_pruning.prune_vgg as modules_vgg
from lrp_pruning.data import NUM_CLASSES
from lrp_pruning.network import Alexnet, ResNet18, ResNet50, Vgg16
from sp_adapters import SPLoRA
from sp_adapters.splora import SPLoRAConv2d


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Structured Pruning of Image Classifiers"
    )

    parser.add_argument(
        "--arch",
        default="resnet50",
        help="model architecture: resnet18, resnet50, vgg16, alexnet",
        choices=["resnet18", "resnet50", "vgg16", "alexnet"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size",
    )
    parser.add_argument("--trialnum", type=int, default=1, help="trial number")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--recovery_epochs",
        type=int,
        default=10,
        help="number of epochs to train to recover accuracy after pruning",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        default=5e-4,
        help="weight decay",
    )
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--norm", action="store_true", help="add normalization")
    parser.add_argument(
        "--resume-from-ckpt",
        type=str,
        default="",
        help="Path to pretrained model",
    )
    parser.add_argument("--train", action="store_true", help="training data")
    parser.add_argument("--prune", action="store_true", help="prune model")
    parser.add_argument(
        "--method-type",
        type=str,
        default="lrp",
        help="model architecture selection",
        choices=["grad", "taylor", "weight", "lrp"],
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=-1,
        help="Limit the number of training batches",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=-1,
        help="Limit the number of testing batches",
    )
    parser.add_argument(
        "--total-pr",
        type=float,
        default=0.9501,
        help="Total pruning rate",
    )
    parser.add_argument(
        "--pr-step",
        type=float,
        default=0.05,
        help="Pruning fraction per step",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="model architecture selection",
        choices=["cifar10", "catsanddogs", "oxfordflowers102"],
    )
    parser.add_argument(
        "--splora",
        action="store_true",
        help="Use Structured Pruning Low-rank Adapter (SPLoRA) for training",
    )
    parser.add_argument(
        "--splora-rank",
        type=int,
        default=16,
        help="Bottleneck dimension of Structured Pruning Low-rank Adapter (SPLoRA).",
    )
    parser.add_argument(
        "--splora-init-range",
        type=float,
        default=1e-3,
        help="Initialisation range of Structured Pruning Low-rank Adapter (SPLoRA).",
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == "__main__":
    args = get_args()

    model = {
        "alexnet": Alexnet,
        "vgg16": Vgg16,
        "resnet18": ResNet18,
        "resnet50": ResNet50,
    }[args.arch.lower()](
        NUM_CLASSES[args.dataset], miniaturize_conv1=(args.dataset == "cifar10")
    )
    if args.splora:
        model = SPLoRA(
            model,
            rank=args.splora_rank,
            init_range=args.splora_init_range,
            replacements=[(torch.nn.Conv2d, SPLoRAConv2d)],
        )

    if args.resume_from_ckpt:
        model.load_state_dict(torch.load(args.resume_from_ckpt))

    if args.cuda:
        model = model.cuda()

    if args.arch.lower() in ["resnet18", "resnet50"]:
        fine_tuner = modules_resnet.PruningFineTuner(args, model)
    elif args.arch.lower() in ["vgg16", "alexnet"]:
        fine_tuner = modules_vgg.PruningFineTuner(args, model)

    if args.train:
        print(
            f"Start training! Dataset: {args.dataset}, Architecture: {args.arch}, Epoch: {args.epochs}"
        )
        fine_tuner.train(epochs=args.epochs)

    if args.prune:
        print(
            f"Start pruning! Dataset: {args.dataset}, Architecture: {args.arch}, Pruning Method: {args.method_type},"
            f" Total Pruning rate: {args.total_pr}, Pruning step: {args.pr_step}"
        )
        fine_tuner.prune()

    sleep(5)  # Allow logging to finalize uploads
