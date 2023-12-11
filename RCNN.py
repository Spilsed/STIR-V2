import os
import random
import statistics
from datetime import datetime
from torchvision import utils

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip
import hashlib
from tqdm import tqdm


class VGGBlock(nn.Module):
    def __init__(self, input, output, num_conv):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(input, output, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            input = output
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 49, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SVM(nn.Module):
    def __init__(self, classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(x)


class RCNN(nn.Module):
    def __init__(self, classes):
        super(RCNN, self).__init__()
        self.vgg = pretrained_custom_vgg16()
        self.svm = SVM(classes)

    def forward(self, x):
        x = self.vgg(x)
        return self.svm(x)


def test_rcnn():
    model = VGG16()
    model.eval()
    model(torch.randn(1, 3, 244, 244))
