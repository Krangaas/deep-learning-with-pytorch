import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from widelenet5 import WideLeNet5
from problem3_utils import *


def main():
    print("PREPROCESSING")
    mnist_data_path = ""
    batch_size = 50

    # Load the pre-trained WideLeNet5 model
    model = WideLeNet5()
    model.load_state_dict(torch.load('model_weights/widelenet5_epoch_Final.pth'))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Load and prepare the MNIST dataset
    test_dataset_large, test_dataset_small = load_and_prepare_data("MNIST", mnist_data_path)
    test_loader_large = DataLoader(test_dataset_large, batch_size=batch_size, shuffle=True)
    test_loader_small = DataLoader(test_dataset_small, batch_size=batch_size, shuffle=True)

    print("TEST")
    test(test_loader_large, model, criterion)
    test(test_loader_small, model, criterion)

if __name__ == "__main__":
    main()
