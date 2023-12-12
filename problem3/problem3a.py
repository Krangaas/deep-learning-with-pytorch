import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from widelenet5 import WideLeNet5
from problem3_utils import *


def main():
    print("PREPROCESSING")
    svhn_data_path = ""
    valid_size = 0.2
    random_seed = 42
    num_epochs = 30
    batch_size = 50
    learning_rate = 0.001
    momentum = 0.8
    weight_decay = 0.001

    print("validation size: %.2f\nepochs: %d\nbatch size: %d\nlearning rate: %.2f\nmomentum: %.2f"
          %(valid_size, num_epochs, batch_size, learning_rate, momentum))

    # Load and prepare the dataset
    train_dataset, test_dataset = load_and_prepare_data("SVHN", svhn_data_path)
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

    # Initialize WideLeNet5 model
    model = WideLeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                momentum=momentum)

    print("TRAINING")
    train(num_epochs, train_loader, valid_loader, model, criterion, optimizer,
          len(valid_idx), len(train_idx), validation_per_epoch=True)

    print("TEST")
    test(test_loader, model, criterion)


if __name__ == "__main__":
    main()
