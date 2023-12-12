import random
from my_resnet18 import Resnet18
from problem2_utils import *


def main():
    # Define the transformation
    # Normalization is calculated from original dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4554421603679657, 0.40690281987190247, 0.389292448759079],
                             std=[0.23302844166755676, 0.21828646957874298, 0.23022505640983582])
    ])

    # Directory containing the images to be transformed
    image_dir = 'train/speed_limits'

    # Load the pre-trained ResNet18 model
    model = Resnet18()
    model.load_state_dict(torch.load('model_weights/resnet18_epoch_10_poisoned.pth'))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Randomly select 10 images from the directory
    sample_image_filenames = random.sample(os.listdir(image_dir), 10)

    # Create and plot saliency and occlusion maps
    saliency_maps(image_dir, sample_image_filenames, model, transform)
    occlusion_sensitivity(image_dir, sample_image_filenames, model, transform)

if __name__ == "__main__":
    main()
