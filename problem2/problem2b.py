import random
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from my_resnet18 import Resnet18
from problem2_utils import *


def main():
    print("PREPROCESSING")
    train_data_path = "train"
    test_data_path = "test"
    valid_size = 0.2
    random_seed = 42
    num_epochs = 10
    batch_size = 30
    learning_rate = 0.1
    momentum = 0.8
    weight_decay = 0.001

    print("validation size: %.2f\nepochs: %d\nbatch size: %d\nlearning rate: %.2f\nmomentum: %.2f"
          %(valid_size, num_epochs, batch_size, learning_rate, momentum))

    # Uncomment the following line(s) to use normalization calculated from original dataset:
    norm = transforms.Normalize(mean=[0.4554421603679657, 0.40690281987190247, 0.389292448759079],
                                std=[0.23302844166755676, 0.21828646957874298, 0.23022505640983582])

    # Uncomment the following line to recalculate the normalization transformation:
    #norm = False

    transform = create_transform(train_data_path, normalization=norm)
    print(transform)

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    train_idx, valid_idx = split_data_idx(train_dataset, valid_size)

    # Shuffle the train and validation data
    np.random.seed(random_seed)
    random.shuffle(train_idx)
    random.shuffle(valid_idx)

    # Initialize sampler objects
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Resnet18 model
    model = Resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                momentum = momentum)

    print("TRAINING")
    train(num_epochs, train_loader, valid_loader, model, criterion, optimizer,
          len(valid_idx), len(train_idx), validation_per_epoch=True)

    print("TEST")
    test(test_loader, model, criterion)

if __name__ == "__main__":
    main()
