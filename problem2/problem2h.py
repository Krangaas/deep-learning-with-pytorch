import random
from my_resnet18 import Resnet18
from problem2_utils import *


def main():
    # Define the transformation
    # Normalization is calculated from clean dataset (after 'generate_fresh_data.py')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44760623574256897, 0.39373499155044556, 0.3757214844226837],
                             std=[0.234161376953125, 0.21670833230018616, 0.22856833040714264])
    ])

    # Directory containing the images to be transformed
    # the script assumes that the data in train/speed_limits is clean
    # i.e. the 'generate_fresh_data.py' script has already been run
    image_dir = 'train/speed_limits'

    # Load the pre-trained ResNet18 model
    model = Resnet18()
    model.load_state_dict(torch.load('model_weights/resnet18_epoch_10.pth'))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Randomly select 10 images from the directory
    sample_image_filenames = random.sample(os.listdir(image_dir), 10)

    # Create and plot saliency and occlusion maps
    saliency_maps(image_dir, sample_image_filenames, model, transform)
    occlusion_sensitivity(image_dir, sample_image_filenames, model, transform)

if __name__ == "__main__":
    main()
