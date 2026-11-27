#assignment7: diffusion
#date: 11-26-26
#name: Joshua wilson
#description: gets the data, downloads the mnist dataset which contains handwritten digit images(0-9)
#converts images into tensors that pytorch can work with
#normalizes pixel values to the range of -1 to 1
# and saves the dataset locally so the other scripts can use it

#gen AI acknowledgement, used gemini to understand the code and make edits and debug

from __future__ import annotations  # Allows more flexible use of type hints throughout the program.

from pathlib import Path  # Imports Path, which makes it easier to create and manage file and folder locations.

import torchvision  # Imports Torchvision, which provides datasets and image-processing tools for PyTorch.
from torchvision import transforms  # Imports image transformations used to prepare images before training.


BASE_DIR = Path(__file__).resolve().parent  # Gets the absolute path of the directory containing this Python file.
DATA_DIR = BASE_DIR / "data"  # Creates the path to a folder named "data" inside the project directory.


def main() -> None:  # Defines the main function that handles downloading and preparing the dataset.

    # ==================== Part 1: Data Download (Provided) ====================
    # Students should run this script first to download MNIST into the data folder.
    # No implementation is required in this file.
    # ==========================================================================

    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Creates the data folder if it does not already exist.
                                                  # parents=True creates any missing parent folders.
                                                  # exist_ok=True prevents an error if the folder already exists.

    # Creates a sequence of image transformations that will be applied to every image.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts each image into a PyTorch tensor and changes pixel values to the range [0, 1].
            transforms.Normalize((0.5,), (0.5,)),  # Normalizes pixel values from [0, 1] to approximately [-1, 1].
        ]
    )

    # Downloads and loads the MNIST training dataset.
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # Specifies the folder where the dataset will be stored.
        train=True,  # Selects the training portion of MNIST (60,000 images).
        download=True,  # Downloads the dataset if it is not already available locally.
        transform=transform,  # Applies the transformations defined above to each image.
    )

    # Downloads and loads the MNIST testing dataset.
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # Specifies the folder where the dataset will be stored.
        train=False,  # Selects the testing portion of MNIST (10,000 images).
        download=True,  # Downloads the dataset if it is not already available locally.
        transform=transform,  # Applies the same transformations used for the training images.
    )

    print(f"MNIST downloaded to: {DATA_DIR}")  # Displays the folder where the dataset was downloaded.
    print(f"Training examples: {len(train_dataset)}")  # Displays the total number of training images.
    print(f"Test examples: {len(test_dataset)}")  # Displays the total number of testing images.


if __name__ == "__main__":  # Checks whether this Python file is being executed directly.
    main()  # Calls the main function if the file is being run directly.
