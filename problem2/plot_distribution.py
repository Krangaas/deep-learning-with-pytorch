import os
import matplotlib.pyplot as plt

def plot_image_distribution(directory, title):
    """ Plot the class distribution for the given directory """
    class_count = {}

    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        class_count[subdirectory] = len(os.listdir(subdirectory_path))

    subdirectories = list(class_count.keys())
    image_counts = list(class_count.values())

    plt.bar(subdirectories, image_counts)
    plt.ylabel("Samples")
    plt.title("%s data class distribution" %title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    train_directory = "train/"
    test_directory = "test/"

    plot_image_distribution(train_directory, "Train")
    plot_image_distribution(test_directory, "Test")

if __name__ == "__main__":
    main()
