import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# fmt: off
cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on


def EfficientNetV2_s(num_classes=10, *args, **kwargs):
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    return model


def EfficientNetV2_m(num_classes=10, *args, **kwargs):
    model = models.efficientnet_v2_m(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    return model


def Vgg16(num_classes=10, *args, **kwargs):
    model = models.__dict__["vgg16"](pretrained=True)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def Alexnet(num_classes=10):
    model = models.__dict__["alexnet"](pretrained=True)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    for param in model.features.parameters():
        param.requires_grad = False
    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, miniaturize_conv1=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = (
            nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            if miniaturize_conv1
            else nn.Conv2d(
                3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def load_imagenet_weights(model, model_name, miniaturize_conv1):
    state_dict = models.__dict__[model_name](pretrained=True).state_dict()
    state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
    if miniaturize_conv1:
        # Downsample conv1: We're handling small input resolutions
        state_dict["conv1.weight"] = F.avg_pool2d(
            state_dict["conv1.weight"], kernel_size=2
        )
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model


def ResNet18(num_classes=10, miniaturize_conv1=True):
    # Specialized ResNet18 for low-res inputs
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, miniaturize_conv1)
    net = load_imagenet_weights(net, "resnet18", miniaturize_conv1)
    return net


def ResNet34(num_classes=10, miniaturize_conv1=True):
    net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, miniaturize_conv1)
    net = load_imagenet_weights(net, "resnet32", miniaturize_conv1)
    return net


def ResNet50(num_classes=10, miniaturize_conv1=True):
    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, miniaturize_conv1)
    net = load_imagenet_weights(net, "resnet50", miniaturize_conv1)
    return net


def ResNet101(num_classes=10, miniaturize_conv1=True):
    net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, miniaturize_conv1)
    net = load_imagenet_weights(net, "resnet101", miniaturize_conv1)
    return net


def ResNet152(num_classes=10, miniaturize_conv1=True):
    net = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, miniaturize_conv1)
    net = load_imagenet_weights(net, "resnet152", miniaturize_conv1)
    return net
