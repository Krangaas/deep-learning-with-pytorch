import os
import gc
import math
from collections import defaultdict

from svhn_loader import load_svhn
from mnist_loader import load_mnist

import torch
from torchvision import transforms
from torch.utils.data import TensorDataset

# Device configuration
device = torch.device('cpu')

def save_state(model_state, epoch):
    """ Save the model weights/state to a .pth file """
    save_filename = "widelenet5_epoch_%s.pth" % epoch
    save_path = os.path.join("./model_weights", save_filename)
    torch.save(model_state, save_path)


def validate(model, data_loader, size, name="Validation"):
    """ Perform validation on the model, given a set of validation datapoints """
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for images, labels in data_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print("Model accuracy on the %s dataset (%d images): %.4f" %(name, size, 100 * correct / total))
    model.train(True)


def test(test_loader, model, criterion):
    """ Perform inference on a set of test data points with the given pretrained model """
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0.0
        total = 0.0

        # Count number of failed predictions against total datapoints per class
        failed_count = {
            0:{"failed":0, "total":0},
            1:{"failed":0, "total":0},
            2:{"failed":0, "total":0},
            3:{"failed":0, "total":0},
            4:{"failed":0, "total":0},
            5:{"failed":0, "total":0},
            6:{"failed":0, "total":0},
            7:{"failed":0, "total":0},
            8:{"failed":0, "total":0},
            9:{"failed":0, "total":0},
        }

        for data, label in test_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(data)

            for predicted, truth in zip(torch.argmax(output,axis = 1), label):
                failed_count[truth.item()]["total"] += 1
                if predicted == truth:
                    correct += 1
                else:
                    failed_count[truth.item()]["failed"] += 1
                total += 1

            # Calculate the test loss
            loss = criterion(output,label)
            test_loss += loss.item() * data.size(0)

        print("Prediction ratio: %d/%d\nTesting Loss: %.4f" %(correct, total, test_loss/len(test_loader)))
        print("Model accuracy on the Test dataset: %f" %(100 * correct / total))
        print(failed_count)
    model.train(True)


def train(num_epochs, train_loader, valid_loader, model, criterion, optimizer,
          valid_size, train_size, validation_per_epoch=False, save_per_epoch=False):
    """
    Train a model given training and validation data, loss and optimizer.
    Saves the model weights per epoch at end of final epoch.
    """
    for epoch in range(num_epochs):
        print("Train step")
        for images, labels in train_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            # Calculate loss
            loss.backward()
            # Optimize
            optimizer.step()


            del images, labels, outputs
            gc.collect()
        print ("Epoch [%d/%d], Training Loss: %.4f" %(epoch+1, num_epochs, loss.item()))

        if save_per_epoch:
            save_state(model.state_dict(), epoch+1)

        if validation_per_epoch:
            print("Validation step")
            validate(model, valid_loader, valid_size)

    # Perform final validation on both the training and validation set
    validate(model, valid_loader, valid_size)
    validate(model, train_loader, train_size, name="Training")
    # Save the final weights
    save_state(model.state_dict(), "Final")
    return


def load_and_prepare_data(dataset, path=''):
    if dataset == "SVHN":
        x_tr, y_tr, x_te, y_te = load_svhn(path)
        # The labels of the SVHN dataset range from 1-10 instead of 0-9,
        # the label ’10’ actually corresponds to the digit 0 in the images.
        # Replace all instances of label ’10’ with ’0’
        y_tr[y_tr == 10] = 0
        y_te[y_te == 10] = 0

    elif dataset == "MNIST":
        x_tr, y_tr, x_te, y_te = load_mnist(path)

    else:
        print("Not such dataset: '%s'. Valid options are: ['SVHN', 'MNIST']" %dataset)
        exit(0)

    # Convert to torch tensor
    x_tr = torch.from_numpy(x_tr).float()
    y_tr = torch.flatten(torch.from_numpy(y_tr).long())
    x_te = torch.from_numpy(x_te).float()
    y_te = torch.flatten(torch.from_numpy(y_te).long())

    # Calculate the mean and std of the train dataset
    mean=torch.mean(x_tr)
    std=torch.std(x_tr)
    norm_transform = transforms.Normalize(mean=mean, std=std)
    print(norm_transform)
    # Apply transform
    x_tr = torch.stack([norm_transform(x) for x in x_tr])
    x_te = torch.stack([norm_transform(x) for x in x_te])

    train_dataset = TensorDataset(x_tr, y_tr)
    test_dataset = TensorDataset(x_te, y_te)

    return train_dataset, test_dataset


def split_data_idx(dataset, valid_size):
    """ Split the dataset into a training and validation set """
    train_data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        train_data_by_class[label.item()].append(idx)

    # Split the train data into train and validation sets per class
    train_data = []
    valid_data = []
    for _, idxs in train_data_by_class.items():
        num_samples = len(idxs)
        num_valid = math.ceil(num_samples * valid_size)
        train_data.extend([idx for idx in idxs[num_valid:]])
        valid_data.extend([idx for idx in idxs[:num_valid]])
    return train_data, valid_data
