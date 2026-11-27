#assignment7: diffusion
#date: 11-26-26
#name: Joshua wilson
#description: generates new images
#loads the trained cnn from the saved checkpoint
# starts completely random noise instead of a handwritten digit
# repeatedly asks the cnn to predict the noise and uses the scheduler to remove some of it
# eventually produces images that hopefully resemble handwritten digits.





# Used Gemini to help implement checkpoint loading and scheduler initialization for inference.
# This script loads the trained CNN, starts with random noise, and uses reverse diffusion
# to gradually generate new images.

from pathlib import Path  # Provides tools for creating and managing file and folder paths.

import torch  # Provides tensor operations, random noise generation, and model inference.
from torchvision.utils import save_image  # Allows generated images to be saved as image files.

from homework7.models.cnn import SimpleCNN  # Imports the CNN architecture used during training.
from homework7.schedulers.simple_noise_scheduler import SimpleNoiseScheduler  # Imports the scheduler that controls the reverse diffusion process.


BASE_DIR = Path(__file__).resolve().parent  # Finds the directory containing this Python file.
DATA_DIR = BASE_DIR / "data"  # Defines the folder containing the trained model checkpoint.
OUT_DIR = BASE_DIR / "out"  # Defines where generated images will be saved.
CHECKPOINT_PATH = DATA_DIR / "simple_cnn_diffusion.pth"  # Defines the location of the previously trained model.


# Edit these values if you want different outputs.
NUM_START_NOISES = 3  # Number of random noise images to generate and use as starting points.
GRID_SIZE = 3  # Determines the dimensions of the generated image grid (3x3 = 9 images).


def denormalize(images: torch.Tensor) -> torch.Tensor:
    # Moves the images to the CPU, converts pixel values from the range [-1, 1] to [0, 1],
    # and clamps any values outside that range so they can be displayed correctly.
    return ((images.cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def run_reverse_diffusion(
    model: SimpleCNN,  # The trained CNN that predicts the noise present in an image.
    scheduler: SimpleNoiseScheduler,  # Controls the mathematical process of removing noise.
    images: torch.Tensor,  # The starting images, which initially contain random noise.
) -> torch.Tensor:

    # Disables gradient calculations because the model is only generating images,
    # not learning or updating its weights.
    with torch.no_grad():

        # Iterates backward through every diffusion timestep.
        # Starts at the highest noise level and works toward timestep zero.
        for step in range(scheduler.num_steps - 1, -1, -1):

            # Creates a tensor containing the current timestep for every image in the batch.
            # torch.full() fills the tensor with the same timestep value.
            timestep = torch.full(
                (images.size(0),),  # Creates one timestep value for each image.
                step,  # The current diffusion timestep.
                device=images.device,  # Places the timestep tensor on the same device as the images.
                dtype=torch.long,  # Stores timestep values as integers.
            )

            # Passes the current noisy images through the CNN.
            # The CNN predicts the noise currently present in each image.
            predicted_noise = model(images)

            # Uses the scheduler and predicted noise to estimate the image
            # at the previous timestep, gradually removing noise.
            images = scheduler.sample_prev_step(images, timestep, predicted_noise)

    # Returns the final images after all reverse diffusion steps are completed.
    return images


def main() -> None:  # Main function responsible for loading the model and generating images.

    OUT_DIR.mkdir(parents=True, exist_ok=True)  # Creates the output folder if it does not already exist.

    # Checks whether a CUDA-compatible GPU is available.
    # Uses the GPU if available; otherwise, falls back to the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Displays which processing device is being used.

    # ==================== Part 4: Load the Trained Model and Scheduler ====================

    # Loads the saved model checkpoint from the training process.
    # map_location=device ensures the checkpoint is loaded onto the selected CPU or GPU.
    # weights_only=True restricts loading to supported weights and basic data structures.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    # Recreates the CNN using the same hidden channel configuration used during training.
    model = SimpleCNN(hidden_channels=checkpoint["hidden_channels"]).to(device)

    # Loads the learned weights and biases from the saved checkpoint into the CNN.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Switches the CNN into evaluation mode.
    # This tells PyTorch the model is being used for inference rather than training.
    model.eval()

    # Recreates the noise scheduler using the same settings that were used during training.
    scheduler = SimpleNoiseScheduler(
        num_steps=checkpoint["num_steps"],  # Restores the number of diffusion timesteps.
        beta_start=checkpoint["beta_start"],  # Restores the starting noise level.
        beta_end=checkpoint["beta_end"],  # Restores the ending noise level.
        device=device,  # Places scheduler tensors on the selected CPU or GPU.
    )

    # ======================================================================================

    # ==================== Part 4: Generate 3 Starting Noise Images ====================

    # Generates three completely random noise images.
    # Each image has 1 channel, a height of 28 pixels, and a width of 28 pixels.
    # These images will serve as the starting points for image generation.
    start_noise = torch.randn(NUM_START_NOISES, 1, 28, 28, device=device)

    # Converts the random noise into a displayable pixel range and saves the three images.
    # nrow=3 places all three starting noise images next to each other.
    save_image(denormalize(start_noise), OUT_DIR / "starting_noise_3.png", nrow=3)

    # ==================== Part 4: Run Reverse Diffusion from 3 Start Points ====================

    # Passes the three random noise images through the reverse diffusion process.
    # The trained CNN predicts noise, and the scheduler gradually removes it.
    generated_from_three = run_reverse_diffusion(model, scheduler, start_noise)

    # Converts the generated images into a displayable pixel range and saves them.
    save_image(denormalize(generated_from_three), OUT_DIR / "generated_from_3_noise.png", nrow=3)

    # ==================== Part 4: Generate a 3x3 Grid of Samples ====================

    # Generates nine new random noise images.
    # GRID_SIZE * GRID_SIZE determines the total number of images (3 * 3 = 9).
    # Each image has 1 channel and dimensions of 28x28 pixels.
    grid_noise = torch.randn(GRID_SIZE * GRID_SIZE, 1, 28, 28, device=device)

    # Runs all nine starting noise images through the reverse diffusion process.
    generated_grid = run_reverse_diffusion(model, scheduler, grid_noise)

    # Converts the generated images into a displayable pixel range and saves them.
    # nrow=GRID_SIZE arranges the nine images into three rows of three.
    save_image(denormalize(generated_grid), OUT_DIR / "generated_grid_3x3.png", nrow=GRID_SIZE)

    # Prints the locations of the saved starting noise and generated images.
    print(f"Saved starting noise images to: {OUT_DIR / 'starting_noise_3.png'}")
    print(f"Saved generated images from 3 start points to: {OUT_DIR / 'generated_from_3_noise.png'}")
    print(f"Saved 3x3 generated grid to: {OUT_DIR / 'generated_grid_3x3.png'}")


# Checks whether this file is being executed directly rather than imported by another Python file.
if __name__ == "__main__":
    main()  # Runs the main function and starts the image generation process.
