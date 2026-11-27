#assignment7: diffusion
#date: 11-26-26
#name: Joshua wilson
#description: teaches the model
#loads handwritten digits from mnist
#uses the noise schedule to add random amounts of noise to the images
# gives the noisy images to the cnn
# compares the cnn predicted noise against the actual noise
# makes adjustments to the cnn weights to make more accurate predictions
# saves the trained model so it can be used later





# Used Gemini to help implement data exploration, noise visualization, and the diffusion training loop.
# This script loads the MNIST dataset, demonstrates noise addition, trains a CNN,
# and saves the trained model and training results.

from pathlib import Path  # Provides tools for creating and managing file and folder paths.

import matplotlib.pyplot as plt  # Used to create graphs, including the training loss graph.
import torch  # Main PyTorch library used for tensors, random noise, and model training.
import torchvision  # Provides image datasets and image-processing utilities.
from torch import nn  # Imports neural network components, including loss functions.
from torch.utils.data import DataLoader  # Loads datasets in batches for more efficient training.
from torchvision import transforms  # Provides tools for modifying and preparing images.

from homework7.models.cnn import SimpleCNN  # Imports the CNN that will learn to predict noise.
from homework7.schedulers.simple_noise_scheduler import SimpleNoiseScheduler  # Imports the class that manages adding noise to images.


BASE_DIR = Path(__file__).resolve().parent  # Finds the directory containing this Python file.
DATA_DIR = BASE_DIR / "data"  # Defines where the MNIST dataset and model checkpoint will be stored.
OUT_DIR = BASE_DIR / "out"  # Defines where generated images, graphs, and training results will be saved.
CHECKPOINT_PATH = DATA_DIR / "simple_cnn_diffusion.pth"  # Defines the location and filename for the saved trained model.


# Edit these values if you want a faster or smaller training run.
# This script assumes MNIST has already been downloaded by 0_download_mnist.py.
# Students can inspect the first 3 images from `dataset` and test noise addition
# with `scheduler.add_noise(...)` before the main training loop if desired.
# NUM_STEPS: total number of diffusion steps in the scheduler.
# BETA_START and BETA_END: the scheduler linearly increases the noise level
# from BETA_START to BETA_END across NUM_STEPS steps.

BATCH_SIZE = 128  # Number of images processed together during one training iteration.
EPOCHS = 40  # Number of times the model will go through the entire training dataset.
LEARNING_RATE = 1e-3  # Controls how large of a step the optimizer takes when updating model weights.
NUM_STEPS = 200  # Total number of possible diffusion timesteps in the noise schedule.
BETA_START = 1e-4  # Amount of noise introduced at the beginning of the diffusion schedule.
BETA_END = 0.02  # Amount of noise introduced at the end of the diffusion schedule.
HIDDEN_CHANNELS = 256  # Controls the number of feature channels used inside the CNN.


def main() -> None:  # Main function that runs the entire data preparation and model training process.

    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Creates the data folder if it does not already exist.
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # Creates the output folder if it does not already exist.

    # Checks whether a CUDA-compatible GPU is available.
    # If one is available, the model will use the GPU; otherwise, it uses the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Displays which processing device will be used.

    # Defines the image transformations that will be applied to MNIST images.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts images into PyTorch tensors with pixel values between 0 and 1.
            transforms.Normalize((0.5,), (0.5,)),  # Normalizes pixel values to approximately -1 to 1.
        ]
    )

    # ==================== Part 1: Data Loading and Exploration ====================

    # Loads the MNIST training dataset into the program.
    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,  # Specifies where the dataset is stored.
        train=True,  # Loads the training portion containing 60,000 images.
        download=False,  # Does not download the dataset because it should already exist.
        transform=transform,  # Applies the image transformations defined above.
    )

    print(f"Dataset size: {len(dataset)}")  # Prints the total number of images in the training dataset.
    print(f"Single image shape: {dataset[0][0].shape}")  # Prints the dimensions of one image tensor.

    # Retrieves the first three images from the dataset and combines them into one tensor.
    first_three_images = torch.stack([dataset[i][0] for i in range(3)])

    # Saves the three images together as a single image file.
    torchvision.utils.save_image(
        first_three_images,  # The three image tensors being saved.
        OUT_DIR / "part1_first_three_clean.png",  # File location and name for the saved images.
        nrow=3,  # Places three images next to each other in one row.
        normalize=True,  # Adjusts image values for proper visual display.
    )
    print(f"Saved first 3 clean images to: {OUT_DIR / 'part1_first_three_clean.png'}")  # Displays where the clean image comparison was saved.

    # ============================================================================

    # Creates a DataLoader that divides the dataset into smaller batches.
    # shuffle=True randomizes the order of images each time the dataset is loaded.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==================== Part 2: Demonstrate Noise Adding ====================

    # Creates the noise scheduler using the settings defined above.
    scheduler = SimpleNoiseScheduler(
        num_steps=NUM_STEPS,  # Total number of diffusion steps.
        beta_start=BETA_START,  # Starting noise level.
        beta_end=BETA_END,  # Ending noise level.
        device=device,  # Stores scheduler tensors on the selected CPU or GPU.
    )

    # Moves the first three clean images onto the selected processing device.
    demo_images = first_three_images.to(device)

    # Generates random Gaussian noise with the same dimensions as the three images.
    demo_noise = torch.randn_like(demo_images)

    # Selects three different timesteps to demonstrate increasing amounts of noise.
    # NUM_STEPS // 100 selects approximately 1% of the total steps.
    # NUM_STEPS // 20 selects approximately 5% of the total steps.
    # NUM_STEPS // 10 selects approximately 10% of the total steps.
    # max() ensures that each selected timestep is at least the specified minimum.
    demo_timesteps = torch.tensor(
        [max(1, NUM_STEPS // 100), max(2, NUM_STEPS // 20), max(3, NUM_STEPS // 10)],
        device=device,  # Creates the timestep tensor on the same device as the scheduler.
    )

    # Adds different amounts of noise to the demonstration images using their selected timesteps.
    noisy_demo_images = scheduler.add_noise(demo_images, demo_timesteps, demo_noise)

    # Combines the original images and noisy images into one tensor.
    # The first three images are clean, and the next three contain noise.
    comparison_grid = torch.cat([demo_images, noisy_demo_images], dim=0)

    # Saves the clean and noisy images together for visual comparison.
    torchvision.utils.save_image(
        comparison_grid.cpu(),  # Moves the images to the CPU because the saving utility uses CPU image data.
        OUT_DIR / "part2_noisy_comparison.png",  # Location and filename of the comparison image.
        nrow=3,  # Displays three images per row, creating a two-row comparison.
        normalize=True,  # Adjusts image values for proper visual display.
    )
    print(
        f"Saved noisy image comparison (timesteps {demo_timesteps.tolist()}) to: "
        f"{OUT_DIR / 'part2_noisy_comparison.png'}"  # Displays the selected timesteps and saved image location.
    )

    # =========================================================================

    # ==================== Part 3: Instantiate Model, Loss, Optimizer ====================

    # Creates the CNN model and moves its parameters onto the selected processing device.
    model = SimpleCNN(hidden_channels=HIDDEN_CHANNELS).to(device)

    # Creates the Adam optimizer, which adjusts the model's weights to reduce prediction errors.
    # model.parameters() gives the optimizer access to the trainable weights and biases.
    # lr controls how large each weight update can be.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Creates the Mean Squared Error loss function.
    # This measures the difference between the actual noise and the noise predicted by the CNN.
    loss_fn = nn.MSELoss()

    # Creates an empty list that will store the average training loss from each epoch.
    loss_history: list[float] = []

    # ==================== Part 3: Training Loop ====================

    # Repeats the training process for the specified number of epochs.
    for epoch in range(1, EPOCHS + 1):

        # Places the model into training mode.
        model.train()

        # Resets the accumulated loss at the beginning of each epoch.
        total_loss = 0.0

        # Goes through the dataset one batch at a time.
        # clean_images contains the images, while _ ignores their digit labels.
        for clean_images, _ in data_loader:

            # Moves the current batch of clean images onto the selected device.
            clean_images = clean_images.to(device)

            # Gets the number of images in the current batch.
            current_batch_size = clean_images.size(0)

            # 1. Sample Gaussian noise

            # Generates random Gaussian noise with the same shape as the clean images.
            # This noise will serve as the correct answer the CNN needs to learn to predict.
            noise = torch.randn_like(clean_images)

            # 2. Sample random timesteps in [0, NUM_STEPS)

            # Randomly selects a diffusion timestep for each image in the current batch.
            # Each image can have a different amount of noise added to it.
            timesteps = torch.randint(0, NUM_STEPS, (current_batch_size,), device=device)

            # 3. Add noise to clean images

            # Uses the scheduler to combine the original images with the generated noise.
            # The selected timestep determines how noisy each image becomes.
            noisy_images = scheduler.add_noise(clean_images, timesteps, noise)

            # 4. Predict the noise with the CNN

            # Passes the noisy images through the CNN.
            # The model attempts to predict the random noise that was originally added.
            predicted_noise = model(noisy_images)

            # 5. Compute MSE loss

            # Compares the CNN's predicted noise against the actual noise.
            # A lower loss means the CNN is becoming better at predicting the noise.
            loss = loss_fn(predicted_noise, noise)

            # 6. Backpropagate and update the model

            # Clears gradients from the previous training iteration.
            # Gradients tell the optimizer how the model's weights should change.
            optimizer.zero_grad()

            # Calculates gradients by working backward through the model.
            # This determines how much each trainable parameter contributed to the error.
            loss.backward()

            # Uses the calculated gradients to update the model's weights and biases.
            optimizer.step()

            # Adds the current batch's loss to the running total.
            # Multiplying by batch size accounts for the number of images in this batch.
            total_loss += loss.item() * current_batch_size

        # Calculates the average loss across the entire training dataset for this epoch.
        epoch_loss = total_loss / len(dataset)

        # Saves the average loss so it can be used later to create a graph.
        loss_history.append(epoch_loss)

        # Prints the current epoch number and its average training loss.
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

    # ===============================================================

    # ==================== Part 3: Save the Model Checkpoint ====================

    # Saves the trained model and its configuration into a checkpoint file.
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # Saves the learned weights and biases of the CNN.
            "num_steps": NUM_STEPS,  # Saves the number of diffusion steps used during training.
            "hidden_channels": HIDDEN_CHANNELS,  # Saves the CNN's hidden channel configuration.
            "beta_start": BETA_START,  # Saves the starting noise level used by the scheduler.
            "beta_end": BETA_END,  # Saves the ending noise level used by the scheduler.
            "uses_timestep_conditioning": False,  # Records that the CNN does not receive timesteps as an input.
        },
        CHECKPOINT_PATH,  # Specifies where the checkpoint file will be saved.
    )

    # Opens a CSV file to store the training loss values.
    # "w" creates a new file or overwrites an existing one.
    # encoding="utf-8" specifies the text encoding used for the file.
    with open(OUT_DIR / "loss_history.csv", "w", encoding="utf-8") as file_obj:

        # Writes the column names at the top of the CSV file.
        file_obj.write("epoch,loss\n")

        # Goes through every recorded loss value.
        # enumerate() provides both the epoch number and its corresponding loss.
        for epoch, loss_value in enumerate(loss_history, start=1):

            # Writes the epoch number and its loss value as a row in the CSV.
            # :.8f formats the loss to eight decimal places.
            file_obj.write(f"{epoch},{loss_value:.8f}\n")

    # Plot training loss curve for Part 5

    # Creates a new figure with a width of 8 inches and a height of 5 inches.
    plt.figure(figsize=(8, 5))

    # Plots the training loss over all epochs.
    # The horizontal axis represents epochs, and the vertical axis represents loss.
    plt.plot(range(1, EPOCHS + 1), loss_history, linewidth=2)

    # Adds a title to the graph.
    plt.title("Training MSE Loss Across Epochs")

    # Labels the horizontal axis.
    plt.xlabel("Epoch")

    # Labels the vertical axis.
    plt.ylabel("MSE Loss")

    # Adds a dashed grid to make the graph easier to read.
    plt.grid(True, linestyle="--", alpha=0.6)

    # Automatically adjusts spacing so the graph elements fit properly.
    plt.tight_layout()

    # Saves the graph as an image file with a resolution of 150 dots per inch.
    plt.savefig(OUT_DIR / "part5_training_loss.png", dpi=150)

    # Closes the figure to free up memory after saving it.
    plt.close()

    # Prints the locations of the saved model, loss history, and training graph.
    print(f"Saved checkpoint to: {CHECKPOINT_PATH}")
    print(f"Saved loss history to: {OUT_DIR / 'loss_history.csv'}")
    print(f"Saved loss plot to: {OUT_DIR / 'part5_training_loss.png'}")


# Checks whether this file is being executed directly rather than imported by another Python file.
if __name__ == "__main__":
    main()  # Runs the main function and starts the entire training process.
