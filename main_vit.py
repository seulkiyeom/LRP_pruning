import argparse
import logging
import math
from datetime import datetime

import torch
import torch_pruning as tp
import torchvision
from sp_adapters import SPLoRA
from sp_adapters.splora import SPLoRALinear, SPLoRAMultiheadAttention
from sp_adapters.torch_pruning import customized_pruners, root_module_types
from tqdm import tqdm

import src.data as datasets
from src.logging import MetricLogger

logging.getLogger("wandb").setLevel(logging.WARNING)


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Structured Pruning of Image Classifiers"
    )

    parser.add_argument(
        "--arch",
        default="vit_b_16",
        help="model architecture: vit_b_16, vit_b_32, vit_l_16, vit_l_32",
        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size",
    )
    parser.add_argument(
        "--init-epochs",
        type=int,
        default=20,
        help="number of epochs to train before pruning",
    )
    parser.add_argument(
        "--recovery-epochs",
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
        "--prune_lr",
        type=float,
        default=0.01,
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
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--resume-from-ckpt",
        type=str,
        default="",
        help="Path to pretrained model",
    )
    parser.add_argument("--train", action="store_true", help="training data")
    parser.add_argument("--prune", action="store_true", help="prune model")
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
        "--target-sparsity",
        type=float,
        default=0.05,
        help="Total pruning rate",
    )
    parser.add_argument(
        "--pr-step",
        type=float,
        default=0.05,
        help="Additional fraction of pruned weights per step",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="model architecture selection",
        choices=[
            "cifar10",
            "cifar100",
            "catsanddogs",
            "oxfordflowers102",
            "stanfordcars",
        ],
    )
    parser.add_argument(
        "--splora",
        action="store_true",
        help="Use Structured Pruning Low-rank Adapter (SPLoRA) for training",
    )
    parser.add_argument(
        "--global-pruning",
        action="store_true",
        help="Whether to rank prunable groups globally",
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

    return args


def train(
    model,
    dataset,
    epochs,
    lr,
    momentum,
    weight_decay,
    limit_train_batches=0,
    description="Train",
    criterion=torch.nn.CrossEntropyLoss(),
    logger=None,
    train_step=0,
    clip_norm=1.0,
):
    model.train()
    num_batches = limit_train_batches if limit_train_batches > 0 else len(dataset)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=num_batches, epochs=epochs
    )

    for _ in tqdm(range(epochs), desc=f"{description} (epoch)"):
        for batch_idx, (data, target) in tqdm(
            enumerate(dataset),
            desc=f"{description} (step)",
            total=num_batches,
            miniters=1,
        ):
            if batch_idx >= num_batches:
                break
            if torch.cuda.is_available():
                data, target = data.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )

            model.zero_grad()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            if logger is not None and train_step % 20 == 0:
                logger.add_scalar("train/step", train_step)
                logger.add_scalar("train/loss", loss.detach().item())
                logger.add_scalar("train/lr", scheduler.get_last_lr()[0])
            train_step += 1

    del optimizer
    del scheduler
    del criterion

    return train_step


def test(
    model,
    dataset,
    limit_test_batches=0,
    description="Test",
    criterion=torch.nn.CrossEntropyLoss(),
):
    model.eval()

    loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(dataset),
            desc=f"{description} (step)",
            total=limit_test_batches if limit_test_batches > 0 else len(dataset),
            miniters=1,
        ):
            if limit_test_batches > 0 and batch_idx >= limit_test_batches:
                break
            if torch.cuda.is_available():
                data, target = data.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )

            pred = model(data)
            loss += criterion(pred, target).item()
            correct += pred.argmax(1).eq(target).sum()
            count += len(target)

    avg_loss = loss / count
    acc = correct / count

    return avg_loss, acc


def count_total_params(model: torch.nn.Module) -> int:
    return int(sum([math.prod(p.shape) for p in model.parameters()]))


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum([math.prod(p.shape) for p in model.parameters() if p.requires_grad]))


def count_fused_params(model: torch.nn.Module) -> int:
    return int(
        sum(
            [
                math.prod(p.shape)
                for n, p in model.named_parameters()
                if "adapt" not in n
            ]
        )
    )


if __name__ == "__main__":
    args = get_args()

    # Prepare logging
    logger = MetricLogger(
        log_dir=f"runs/{datetime.now().isoformat()}",
        loggers={"wandb", "stdout"},
        args=args,
    )
    try:
        # Ensure reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        logger.add_scalar("seed", args.seed)

        # Prepare dataset
        args.dataset = args.dataset.lower()
        image_size = 224

        train_ds, test_ds = {
            "cifar10": datasets.get_cifar100,
            "catsanddogs": datasets.get_catsanddogs,
            "cifar100": datasets.get_cifar100,
            "oxfordflowers102": datasets.get_oxfordflowers102,
            "stanfordcars": datasets.get_stanfordcars,
            "imagenet": datasets.get_imagenet,
        }[args.dataset](image_size=image_size)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        num_classes = datasets.NUM_CLASSES[args.dataset]

        example_input = torch.randn(1, 3, image_size, image_size)

        # Prepare model
        model = {
            "vit_b_16": torchvision.models.vit_b_16,
            "vit_b_32": torchvision.models.vit_b_32,
            "vit_l_16": torchvision.models.vit_l_16,
            "vit_l_32": torchvision.models.vit_l_32,
        }[args.arch.lower()](
            pretrained=True,
        )

        arch = args.arch
        if args.splora:
            arch += "_splora-" + str(args.splora_rank)

        if args.splora:
            model = SPLoRA(
                model,
                rank=args.splora_rank,
                init_range=args.splora_init_range,
                replacements=[
                    (torch.nn.MultiheadAttention, SPLoRAMultiheadAttention),
                    (torch.nn.Linear, SPLoRALinear),
                    # (torch.nn.Conv2d, SPLoRAConv2d), # Skip conv
                ],
            )

        # Reinitialize classifier
        model.heads.head = torch.nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=num_classes,
            bias=True,
        ).to(model.heads.head.weight.device)

        # Skip conv layer - we don't have an efficient adapter for
        # large-kernel, large-stride convs
        model.conv_proj.weight.requires_grad = False

        if args.resume_from_ckpt:
            model.load_state_dict(torch.load(args.resume_from_ckpt))

        if torch.cuda.is_available():
            model = model.cuda()
            example_input = example_input.cuda(non_blocking=True)

        # Prepare pruner
        imp = tp.importance.MagnitudeImportance(p=2)

        round_to = None
        if isinstance(model, torchvision.models.vision_transformer.VisionTransformer):
            round_to = model.encoder.layers[0].num_heads  # model-specific restriction

        iterative_steps = round((1 - args.target_sparsity) / args.pr_step)

        pruner = tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_input,
            importance=imp,  # Importance Estimator
            global_pruning=args.global_pruning,  # Please refer to Page 9 of https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf
            ch_sparsity=0.95,  # global sparsity for all layers
            # ch_sparsity_dict = {model.conv1: 0.2}, # manually set the sparsity of model.conv1
            iterative_steps=iterative_steps,  # number of steps to achieve the target ch_sparsity.
            ignored_layers=[model.heads.head],  # ignore final linear classifier
            round_to=round_to,  # round channels
            customized_pruners=customized_pruners,
            root_module_types=root_module_types,
        )

        # Make initial adaptation to model
        base_macs, _ = tp.utils.count_ops_and_params(model, example_input)
        base_fused_params = count_fused_params(model)
        base_trainable_paramse = count_trainable_params(model)
        logger.add_scalar("train/macs", base_macs)
        logger.add_scalar("train/fused_params", count_fused_params(model))
        logger.add_scalar("train/trainable_params", count_trainable_params(model))
        logger.add_scalar("train/total_params", count_total_params(model))
        logger.add_scalar("train/sparsity", 1.0)

        prev_train_step = 0

        if args.train:
            # Freeze all but the untrained head
            for param in model.parameters():
                param.requires_grad = False

            for param in model.heads.head.parameters():
                param.requires_grad = True

            # Train the head
            prev_train_step = train(
                model=model,
                dataset=train_loader,
                epochs=max(1, args.init_epochs // 2),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                limit_train_batches=args.limit_train_batches,
                description="Train head",
                logger=logger,
            )
            test_loss, test_acc = test(
                model, test_loader, limit_test_batches=args.limit_test_batches
            )
            logger.add_scalar("test/acc", test_acc)
            logger.add_scalar("test/loss", test_loss)
            logger.add_scalar("test/macs", base_macs)
            logger.add_scalar("test/fused_params", count_fused_params(model))
            logger.add_scalar("test/trainable_params", count_trainable_params(model))
            logger.add_scalar("test/total_params", count_total_params(model))
            logger.add_scalar("test/sparsity", 1.0)

            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True

            model.conv_proj.weight.requires_grad = False

            prev_train_step = train(
                model=model,
                dataset=train_loader,
                epochs=max(1, args.init_epochs // 2),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                limit_train_batches=args.limit_train_batches,
                description="Train whole network",
                train_step=prev_train_step,
                logger=logger,
            )

            test_loss, test_acc = test(
                model, test_loader, limit_test_batches=args.limit_test_batches
            )
            logger.add_scalar("test/acc", test_acc)
            logger.add_scalar("test/loss", test_loss)
            logger.add_scalar("test/macs", base_macs)
            logger.add_scalar("test/fused_params", count_fused_params(model))
            logger.add_scalar("test/trainable_params", count_trainable_params(model))
            logger.add_scalar("test/total_params", count_total_params(model))
            logger.add_scalar("test/sparsity", 1.0)
            torch.save(
                model.state_dict(),
                f"{logger.log_dir}/{arch}_{args.dataset}_100%.pth",
            )

        # Perform iterative pruning and finetuning
        if args.prune:
            fused_params = count_fused_params(model)
            while count_fused_params(model) / base_fused_params > args.target_sparsity:
                pruner.step()
                # Pruner sets updated requires_grad of updated weight to True
                model.conv_proj.weight.requires_grad = False

                # ViT relies on the hidden_dim attribute for forwarding, so we have to modify this variable after pruning
                if isinstance(
                    model, torchvision.models.vision_transformer.VisionTransformer
                ):
                    model.hidden_dim = model.conv_proj.out_channels

                prev_train_step = train(
                    model=model,
                    dataset=train_loader,
                    epochs=args.recovery_epochs,
                    lr=args.prune_lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    limit_train_batches=args.limit_train_batches,
                    description=f"Training after pruning ({(count_fused_params(model) / base_fused_params):.2f})",
                    logger=logger,
                    train_step=prev_train_step,
                )

                logger.add_scalar(
                    "train/macs", tp.utils.count_ops_and_params(model, example_input)[0]
                )
                logger.add_scalar("train/fused_params", count_fused_params(model))
                logger.add_scalar(
                    "train/trainable_params", count_trainable_params(model)
                )
                logger.add_scalar("train/total_params", count_total_params(model))
                logger.add_scalar(
                    "train/sparsity", count_fused_params(model) / base_fused_params
                )

                test_loss, test_acc = test(
                    model,
                    test_loader,
                    limit_test_batches=args.limit_test_batches,
                    description=f"Testing after pruning ({(count_fused_params(model) / base_fused_params):.2f})",
                )
                logger.add_scalar("test/loss", test_loss)
                logger.add_scalar("test/acc", test_acc)
                logger.add_scalar(
                    "test/macs", tp.utils.count_ops_and_params(model, example_input)[0]
                )
                logger.add_scalar("test/fused_params", count_fused_params(model))
                logger.add_scalar(
                    "test/trainable_params", count_trainable_params(model)
                )
                logger.add_scalar("test/total_params", count_total_params(model))
                logger.add_scalar(
                    "test/sparsity", count_fused_params(model) / base_fused_params
                )
                torch.save(
                    model.state_dict(),
                    f"{logger.log_dir}/{arch}_{args.dataset}_{round(fused_params / base_fused_params * 100)}%.pth",
                )

    except KeyboardInterrupt as e:
        logger.close(1)
        raise e
