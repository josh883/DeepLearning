#explore file
#joshua wilson
#10-9-26
#summary:Inspects and visualizes the Oxford-IIIT Pet dataset before training.
#Checks dataset size (number of train and test samples).
#Inspects dimensions, labels, and mask values of the raw PIL images.
#Generates and saves a 3×2 grid comparing the first three training images with their ground-truth segmentation masks (figures/sample_images.png).
#Passes a batch through get_loaders() to verify processed tensor shapes ([16, 3, 128, 128] for images, [16] for labels, [16, 128, 128] for masks).

# GenAI acknowledgment: made using claude + gemini for debugging.

import os  # Import operating system module to manage directories
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images and masks
import numpy as np  # Import NumPy for array operations and unique value checking
from data import (  # Import raw dataset loader, batch loader, and class mappings
    CLASS_NAMES,
    get_loaders,
    get_raw_datasets,
)#class names and raw data

os.makedirs(
    "figures", exist_ok=True
)  # Create the 'figures/' directory if it doesn't exist yet

# ---- 2. Explore the dataset ----
train_dataset, test_dataset = (
    get_raw_datasets()
)  # Load the raw PIL image datasets for training and testing
print(
    "Training samples:", len(train_dataset)
)  # Output the total number of training images
print(
    "Testing samples: ", len(test_dataset)
)  # Output the total number of testing images

image, (label, mask) = (
    train_dataset[0]
)  # Extract the very first sample: PIL image, class integer label, and mask PIL image
print(
    "First image size (width, height):", image.size
)  # Print dimensions (width, height) of the raw PIL image
print(
    "First image label:", label, f"({CLASS_NAMES[label]})"
)  # Print numerical label alongside its mapped class name (Cat/Dog)
print(
    "First mask size (width, height): ", mask.size
)  # Print dimensions (width, height) of the raw PIL segmentation mask
print(
    "First mask unique values:", np.unique(np.array(mask))
)  # Convert mask to NumPy array and print its distinct pixel class values

# ---- 3. Visualize first three training images beside their masks ----
fig, axes = plt.subplots(
    3, 2, figsize=(6, 9)
)  # Create a 3x2 grid of subplots with a specified figure size
for i in range(3):  # Loop through the first 3 dataset samples
  img, (lbl, msk) = train_dataset[i]  # Retrieve image, label, and mask for index i
  axes[i, 0].imshow(img)  # Render the original PIL image on the left subplot
  axes[i, 0].set_title(
      f"Image {i} ({CLASS_NAMES[lbl]})"
  )  # Set title showing sample index and class name
  # raw masks use values 1, 2, 3, so fix the color range to keep colors consistent
  axes[i, 1].imshow(
      np.array(msk), cmap="viridis", vmin=1, vmax=3
  )  # Render mask on right subplot with locked color normalization range
  axes[i, 1].set_title(
      f"Mask {i}"
  )  # Set title for the right mask subplot
  axes[i, 0].axis("off")  # Hide axis ticks and boundaries for the image
  axes[i, 1].axis("off")  # Hide axis ticks and boundaries for the mask
plt.tight_layout()  # Automatically adjust layout spacing to avoid overlapping titles
plt.savefig(
    "figures/sample_images.png", dpi=150
)  # Save the resulting figure grid to disk at 150 DPI
plt.close()  # Close the plot memory context to free up memory resources

# ---- 4. Inspect one processed batch ----
train_loader, test_loader = get_loaders(
    batch_size=16
)  # Create PyTorch DataLoaders with a batch size of 16
images, (labels, masks) = next(
    iter(train_loader)
)  # Fetch one single processed mini-batch from the training loader
print(
    "images shape:", images.shape
)  # Print image batch tensor shape [batch_size, channels, height, width]
print("labels shape:", labels.shape)  # Print class label batch tensor shape [batch_size]
print(
    "masks shape: ", masks.shape
)  # Print mask batch tensor shape [batch_size, height, width]