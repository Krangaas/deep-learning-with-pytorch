import os
import gc
import math
import numpy as np

import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import DataLoader

from captum.attr import Occlusion, Saliency
from captum.attr import visualization as viz

from PIL import Image

# Device configuration
device = torch.device('cpu')


def save_state(model_state, epoch):
    """ Save the model weights/state to a .pth file """
    save_filename = "resnet18_epoch_%s.pth" % epoch
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
            4:{"failed":0, "total":0}
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


def create_transform(data_path, normalization=False):
    """
    Instantiate and return a transformation object.
    Normalization is calculated from input data, unless given as input.
    """
    if not normalization:
        train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transforms.ToTensor())
        # Calculate the mean and std of the train dataset
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        mean = data[0].mean(dim=(0, 2, 3))
        std = data[0].std(dim=(0, 2, 3))
        normalization = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    # Define transform
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalization
    ])

    return transform

def create_crop_transform(data_path, normalization=False, mode=None):
    """
    Instantiate and return a transformation object.
    Normalization is calculated from cropped input data, unless given as input.
    Valid transformation modes after cropping are ('pad', 'resize').
    """
    if not normalization:
        train_dataset = torchvision.datasets.ImageFolder(root=data_path,
                                                         transform=transforms.Compose([
                                                            transforms.CenterCrop((204,204)),
                                                            transforms.ToTensor()])
                                                        )
        # Calculate the mean and std of the train dataset asfter cropping
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        mean = data[0].mean(dim=(0, 2, 3))
        std = data[0].std(dim=(0, 2, 3))
        normalization = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    if mode == "pad":
        mode = transforms.Pad(padding=10)
    else:
        mode = transforms.Resize((224,224))

    # Define transform
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop((204,204)),
            mode,
            transforms.ToTensor(),
            normalization
    ])

    return transform


def split_data_idx(dataset, valid_size):
    """ Split the dataset into a training and validation set """
    train_data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        train_data_by_class[label].append(idx)

    # Split the train data into train and validation sets per class
    train_data = []
    valid_data = []
    for _, idxs in train_data_by_class.items():
        num_samples = len(idxs)
        num_valid = math.ceil(num_samples * valid_size)
        train_data.extend([idx for idx in idxs[num_valid:]])
        valid_data.extend([idx for idx in idxs[:num_valid]])

    return train_data, valid_data


def saliency_maps(image_dir, sample_image_filenames, model, transform):
    """ Calculate and plot saliency maps for the given sample images, using the given pretrained model """
    original_images = []
    transformed_images = []

    # Create a transform without normalization step
    # Original images will only be used for plotting, and so normalization is uneccesary
    transform_no_norm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    # Loop over the sample images.
    # For each image, save one normalized version and one un-normalized
    for filename in sample_image_filenames:
        image = Image.open(os.path.join(image_dir, filename))

        # Apply transformation to image with normalization
        # Reshape to fit expected model input shape.
        # I.e. set batch size to 1, as the model expects the input to be a batch of images
        transformed_image = transform(image).reshape(1, 3, 224, 224)
        transformed_image.requires_grad_()

        # Apply transform to original image without normalization
        image = transform_no_norm(image)

        original_images.append(image)
        transformed_images.append(transformed_image)

    # Create a saliency class object that wraps around the model to catch the output
    saliency = Saliency(model)

    # For each image, plot the original image along with the corresponding saliency map
    for tr_img, org_img in zip(transformed_images, original_images):
        attributions_sal = saliency.attribute(tr_img, target=1)

        fig, ax = viz.visualize_image_attr_multiple(
            np.transpose(attributions_sal.squeeze().detach().numpy(), (1,2,0)),
            np.transpose(org_img.squeeze().detach().numpy(), (1,2,0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            ["Original", "Positive attribution"],
            fig_size=(6,6)
        )


def occlusion_sensitivity(image_dir, sample_image_filenames, model, transform, occ_params=False):
    """
    Calculate and plot occlusion images for the given sample images and the pretrained model.
    By default, this method will calculate both positive and negative attribution plots.
    """
    original_images = []
    transformed_images = []
    if not occ_params:
        occ_params = {
            "target": 1,         # class index to compute difference against. In this case it should be 1 (speed_limits)
            "stride": (3,6,6),   # occlusion window step size the channel stride must be <= the window channel size to cover all parts of the image are covered
            "window": (3,10,10), # occlusion window size
            "baseline": 0.5,     # baseline value to replace features
                                 # the images are normalized between 0 and 1, so 0.5 corresponds to gray
            "methods": ["original_image", "heat_map", "heat_map"], # output image methods
            "signs": ["all", "positive", "negative"],              # output image sign
            "titles": ["Original", "Positive attribution", "Negative attribution"], # image titles
            "figsize": (6,6)                                                        # figure size
        }

    # Create a transform without normalization step
    # Original images will only be used for plotting, and so normalization is uneccesary
    transform_no_norm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    # Loop over the sample images.
    # For each image, save one normalized version and one un-normalized
    for filename in sample_image_filenames:
        image = Image.open(os.path.join(image_dir, filename))

        # Apply transformation to image with normalization
        # Reshape to fit expected model input shape.
        # I.e. set batch size to 1, as the model expects the input to be a batch of images
        transformed_image = transform(image).reshape(1, 3, 224, 224)

        # Apply transform to original image without normalization
        image = transform_no_norm(image)

        original_images.append(image)
        transformed_images.append(transformed_image)

    # Create an occlusion class object that wraps around the model to catch the output
    occlusion = Occlusion(model)

    # For each image, plot the original image along with the corresponding positive and negative attribution maps
    for tr_img, org_img in zip(transformed_images, original_images):
        attributions_occ = occlusion.attribute(
            tr_img,
            target=occ_params["target"],
            strides=occ_params["stride"],
            sliding_window_shapes=occ_params["window"],
            baselines=occ_params["baseline"],
            show_progress=True)

        fig, ax = viz.visualize_image_attr_multiple(
            np.transpose(attributions_occ.squeeze().detach().numpy(), (1,2,0)),
            np.transpose(org_img.squeeze().detach().numpy(), (1,2,0)),
            occ_params["methods"],
            occ_params["signs"],
            show_colorbar=True,
            outlier_perc=2,
            titles=occ_params["titles"],
            fig_size=occ_params["figsize"])
    return
