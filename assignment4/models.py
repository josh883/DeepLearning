#models file
#joshua wilson
#10-9-26
#Defines the three neural network architectures required for the assignment.
#A custom 3-layer CNN ending with global average pooling and a linear classifier
#build_resnet18(): Instantiates PyTorch's resnet18(weights=None) from scratch and replaces the final fc layer to output 2 class logits.
#DoubleConv & SmallUNet: Implements an encoder-decoder segmentation architecture with skip connections using torch.cat(dim=1) to output 3 class logits per pixel.



# GenAI acknowledgment: Used Claude (Anthropic) to help draft the model + gemini for debugging
# definitions (CNN, U-Net, ResNet-18 builder). I reviewed and tested them myself. 

import torch  # Import PyTorch core library for tensors and neural networks
import torch.nn as nn  # Import neural network modules and layer definitions
from torchvision.models import (
    resnet18,
)  # Import pre-defined ResNet-18 model architecture from torchvision


# Define the custom 3-layer Convolutional Neural Network for binary classification
class SimpleCNN(nn.Module):

  def __init__(self):
    super().__init__()  # Initialize the parent nn.Module class
    self.features = nn.Sequential(  # Define feature extractor pipeline sequentially
        nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        ),  # 1st Conv: 3 RGB channels -> 16 feature maps (128x128)
        nn.ReLU(),  # Element-wise ReLU activation function
        nn.MaxPool2d(2),  # Downsample spatial dimensions by factor of 2 (128x128 -> 64x64)
        nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        ),  # 2nd Conv: 16 -> 32 feature maps (64x64)
        nn.ReLU(),  # Element-wise ReLU activation function
        nn.MaxPool2d(2),  # Downsample spatial dimensions by factor of 2 (64x64 -> 32x32)
        nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        ),  # 3rd Conv: 32 -> 64 feature maps (32x32)
        nn.ReLU(),  # Element-wise ReLU activation function
        nn.AdaptiveAvgPool2d((
            1,
            1,
        )),  # Global average pooling: reduces spatial 32x32 to 1x1 vector per channel
    )
    self.fc = nn.Linear(
        64, 2
    )  # Fully connected layer: maps 64 pooled features to 2 class logits (Cat/Dog)

  def forward(self, x):
    x = self.features(
        x
    )  # Pass input batch [B, 3, 128, 128] through feature extraction -> [B, 64, 1, 1]
    x = torch.flatten(
        x, start_dim=1
    )  # Flatten spatial dimensions into a 1D vector per sample -> [B, 64]
    return self.fc(x)  # Compute output logits -> [B, 2]


# Helper function to construct ResNet-18 with randomly initialized weights
def build_resnet18():
  model = resnet18(
      weights=None
  )  # Load standard ResNet-18 without pre-trained weights
  model.fc = nn.Linear(
      model.fc.in_features, 2
  )  # Replace final FC layer (1000 classes -> 2 classes) before optimizer setup
  return model  # Return adjusted ResNet-18 instance


# Reusable convolutional block used throughout the U-Net architecture
class DoubleConv(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()  # Initialize the parent nn.Module class
    self.block = nn.Sequential(  # Group dual 3x3 convolutions with ReLU activations
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        ),  # 1st Conv: padding=1 preserves height & width
        nn.ReLU(),  # 1st ReLU activation
        nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        ),  # 2nd Conv: preserves feature dimensions
        nn.ReLU(),  # 2nd ReLU activation
    )

  def forward(self, x):
    return self.block(x)  # Pass input through the double convolution block


# Custom U-Net model implementation for pixel-level semantic segmentation
class SmallUNet(nn.Module):

  def __init__(self):
    super().__init__()  # Initialize parent nn.Module class
    self.enc1 = DoubleConv(
        3, 16
    )  # 1st Encoder stage: RGB input -> 16 feature channels
    self.enc2 = DoubleConv(
        16, 32
    )  # 2nd Encoder stage: 16 -> 32 feature channels
    self.pool = nn.MaxPool2d(
        2
    )  # Shared max pooling layer: halves spatial height & width
    self.bottleneck = DoubleConv(
        32, 64
    )  # Bottleneck layer: connects encoder and decoder at lowest resolution

    self.up1 = nn.ConvTranspose2d(
        64, 32, kernel_size=2, stride=2
    )  # 1st Up-conv: doubles spatial resolution (32x32 -> 64x64) and reduces channels to 32
    self.dec1 = DoubleConv(
        64, 32
    )  # 1st Decoder block: processes concatenated features (32 upsampled + 32 skip = 64)
    self.up2 = nn.ConvTranspose2d(
        32, 16, kernel_size=2, stride=2
    )  # 2nd Up-conv: doubles spatial resolution (64x64 -> 128x128) and reduces channels to 16
    self.dec2 = DoubleConv(
        32, 16
    )  # 2nd Decoder block: processes concatenated features (16 upsampled + 16 skip = 32)
    self.out = nn.Conv2d(
        16, 3, kernel_size=1
    )  # Final 1x1 Conv: maps 16 features to 3 output class logits per pixel

  def forward(self, x):
    e1 = self.enc1(x)  # Compute 1st encoder block output -> shape: [B, 16, 128, 128]
    e2 = self.enc2(
        self.pool(e1)
    )  # Pool e1 to [B, 16, 64, 64] and pass to 2nd block -> shape: [B, 32, 64, 64]
    b = self.bottleneck(
        self.pool(e2)
    )  # Pool e2 to [B, 32, 32, 32] and pass to bottleneck -> shape: [B, 64, 32, 32]

    d1 = self.up1(b)  # Upsample bottleneck output -> shape: [B, 32, 64, 64]
    d1 = self.dec1(
        torch.cat([d1, e2], dim=1)
    )  # Concatenate with skip connection e2 on channel dim (32+32=64) -> output: [B, 32, 64, 64]
    d2 = self.up2(d1)  # Upsample decoder output -> shape: [B, 16, 128, 128]
    d2 = self.dec2(
        torch.cat([d2, e1], dim=1)
    )  # Concatenate with skip connection e1 on channel dim (16+16=32) -> output: [B, 16, 128, 128]
    return self.out(
        d2
    )  # Map to 3 class channels per pixel via 1x1 conv -> shape: [B, 3, 128, 128]


if __name__ == "__main__":
  # Shape sanity check with a real batch from the data loader
  from data import get_loaders  # Import batch generator function from data script

  torch.manual_seed(0)  # Set random seed for consistent testing checks
  device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
  )  # Select CUDA GPU if available, otherwise CPU
  print("Device:", device)  # Print active device

  train_loader, _ = get_loaders(
      batch_size=16
  )  # Load training data generator with batch size 16
  images, (labels, masks) = next(
      iter(train_loader)
  )  # Unpack a test batch of images and targets
  images = images.to(device)  # Transfer input image tensor to active device

  # Iterate through all 3 model instances to check shapes and parameter counts
  for name, model in [
      ("CNN", SimpleCNN()),
      ("ResNet-18", build_resnet18()),
      ("U-Net", SmallUNet()),
  ]:
    model = model.to(device)  # Move model weights to selected device
    out = model(images)  # Run forward pass with test batch
    n_params = sum(
        p.numel() for p in model.parameters()
    )  # Count total trainable/non-trainable parameters
    print(
        f"{name}: output shape {tuple(out.shape)}, parameters {n_params:,}"
    )  # Output test verification results