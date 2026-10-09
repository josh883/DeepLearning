#train file
#joshua wilson
#10-9-26
#summary:Handles the end-to-end training process for all three models.
#enforces seed reproducibility (torch.manual_seed(0)) and device management (CUDA GPU or CPU).
#Routes targets based on model type: labels (Cat vs. Dog) for CNN & ResNet-18, masks for U-Net.
#Executes the standard PyTorch training loop (zero_grad, forward pass, CrossEntropyLoss, backward, step).
#Tracks and computes average loss per epoch.
#Saves trained weights (*.pt) and exports per-epoch loss histories to JSON (losses_*.json).


# GenAI acknowledgment: Used Claude to help draft this training
# script (shared training loop, loss saving). I reviewed and tested it myself. + gemini for debugging

import argparse  # Import command-line argument parser module
import json  # Import JSON encoder/decoder for saving loss history lists
import time  # Import time module to measure per-epoch training duration
import torch  # Import PyTorch core library
import torch.nn as nn  # Import PyTorch neural network modules (e.g. CrossEntropyLoss)
from data import get_loaders  # Import dataloader generator from dataset module
from models import SimpleCNN, SmallUNet, build_resnet18  # Import model architectures

SEED = 0  # Global seed value for reproducible weight initialization
BATCH_SIZE = 16  # Mini-batch size for training dataloader
LR = 0.001  # Fixed learning rate for Adam optimizer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Select CUDA GPU if available, else fall back to CPU


def train_model(model, train_loader, target_type, epochs, name):
  """target_type: 'labels' for the classifiers, 'masks' for the U-Net."""
  model = model.to(device)  # Move model weights to the target computing device
  criterion = (
      nn.CrossEntropyLoss()
  )  # Define loss function (expects raw logits, no internal softmax needed)
  optimizer = torch.optim.Adam(
      model.parameters(), lr=LR
  )  # Instantiate separate Adam optimizer for this model instance
  losses = []  # List to store average loss recorded at each epoch
  model.train()  # Set model to training mode (enables dropout/batchnorm behavior if present)
  for epoch in range(epochs):  # Outer loop iterating through required epochs
    start = time.time()  # Track epoch execution start time
    running, seen = 0.0, 0  # Initialize cumulative loss and processed sample counter
    for images, (labels, masks) in train_loader:  # Iteration through mini-batches
      images = images.to(device)  # Transfer input image batch tensor to device
      targets = (labels if target_type == "labels" else masks).to(
          device
      )  # Route target tensor (labels or masks) and transfer to device
      optimizer.zero_grad()  # Clear existing parameter gradients before backward pass
      outputs = model(images)  # Run forward pass to compute output logits
      loss = criterion(
          outputs, targets
      )  # Calculate CrossEntropyLoss against chosen targets
      loss.backward()  # Compute gradients via backpropagation
      optimizer.step()  # Update model weights based on computed gradients
      running += (
          loss.item() * images.size(0)
      )  # Accumulate total loss weighted by current batch size
      seen += images.size(0)  # Increment count of processed samples
    avg = running / seen  # Compute weighted average loss over the entire epoch
    losses.append(avg)  # Store epoch average loss value in list
    print(
        f"[{name}] epoch {epoch + 1}/{epochs}  loss {avg:.4f} "
        f" ({time.time() - start:.1f}s)"
    )  # Output progress and timing summary
  return losses  # Return complete epoch loss history list


if __name__ == "__main__":
  parser = argparse.ArgumentParser()  # Initialize argument parser for script options
  parser.add_argument(
      "--model", choices=["cnn", "resnet", "unet", "all"], default="all"
  )  # CLI arg to pick specific model or run all
  parser.add_argument(
      "--epochs", type=int, default=1
  )  # CLI arg to set total training epochs (default 1)
  args = parser.parse_args()  # Parse command-line flags

  train_loader, _ = get_loaders(
      batch_size=BATCH_SIZE
  )  # Load dataset training batch generator

  # Map model names to their respective factory functions and target target_types
  jobs = {
      "cnn": (SimpleCNN, "labels"),
      "resnet": (build_resnet18, "labels"),
      "unet": (SmallUNet, "masks"),
  }
  names = (
      list(jobs) if args.model == "all" else [args.model]
  )  # Select list of model keys to train

  for name in names:  # Loop through each model selected for execution
    torch.manual_seed(SEED)  # Reset manual seed before building each model to ensure consistency
    build, target_type = jobs[name]  # Unpack builder function and target selection string
    model = build()  # Instantiate model architecture with default initialization
    losses = train_model(
        model, train_loader, target_type, args.epochs, name
    )  # Train model and get loss history
    torch.save(
        model.state_dict(), f"{name}.pt"
    )  # Save trained model weights to PyTorch checkpoint file
    with open(f"losses_{name}.json", "w") as f:
      json.dump(
          losses, f
      )  # Export epoch loss list to JSON file for easy plotting in evaluation
    print(f"Saved {name}.pt and losses_{name}.json\n")  # Log save completion message