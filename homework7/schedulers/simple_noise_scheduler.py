#assignment7: diffusion
#date: 11-26-26
#name: Joshua wilson
#description: handles the noise math
# defines how much noise gets added to an image at different timesteps
#contains the forward process which adds random noise to clean images
#contains the reverse process, which uses predicted noise to gradually remove noise
# think of this file as the mathematical engine behind the diffusion process

#gen AI acknowledgement gemini used to understand the code

from __future__ import annotations  # Allows type hints to reference classes and types before they are fully defined.

import torch  # Imports PyTorch, which provides tensor operations and mathematical functions used by the scheduler.


class SimpleNoiseScheduler:
    """A minimal DDPM-style linear noise scheduler.

    Parameters:
    - num_steps: total number of diffusion steps.
    - beta_start: noise level at the first step.
    - beta_end: noise level at the last step.
    - device: CPU or GPU device used to store the schedule tensors.
    """

    def __init__(
        self,
        num_steps: int = 200,  # Number of steps used to gradually add or remove noise.
        beta_start: float = 1e-4,  # Starting amount of noise added during the diffusion process.
        beta_end: float = 0.02,  # Ending amount of noise added during the diffusion process.
        device: str | torch.device = "cpu",  # Determines whether calculations happen on the CPU or GPU.
    ) -> None:
        self.num_steps = num_steps  # Stores the total number of diffusion steps.
        self.device = torch.device(device)  # Converts the device argument into a PyTorch device object.

        # Creates evenly spaced beta values between beta_start and beta_end.
        # Each beta represents how much noise is introduced at a particular step.
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)

        # Calculates alpha by subtracting beta from 1.
        # Alpha represents how much of the original image information is preserved at each step.
        self.alphas = 1.0 - self.betas

        # Calculates the cumulative product of all alpha values.
        # This tells us how much original image information remains after multiple noise steps.
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        clean_images: torch.Tensor,  # The original images before any noise has been added.
        timesteps: torch.Tensor,  # The diffusion step assigned to each image in the batch.
        noise: torch.Tensor | None = None,  # Optional noise tensor; if none is provided, random noise is generated.
    ) -> torch.Tensor:
        # Checks whether noise was provided by the caller.
        if noise is None:
            noise = torch.randn_like(clean_images)  # Generates random noise with the same shape as the original images.

        # Converts timestep values to integers and flattens them into a one-dimensional tensor.
        timesteps = timesteps.long().view(-1)

        # Retrieves the cumulative alpha value for each image's timestep.
        # Reshaping allows the values to broadcast across the image's height, width, and channels.
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        # Combines the original image and random noise.
        # The first term controls how much original image remains.
        # The second term controls how much random noise is added.
        return torch.sqrt(alpha_bar_t) * clean_images + torch.sqrt(1.0 - alpha_bar_t) * noise

    def sample_prev_step(
        self,
        noisy_images: torch.Tensor,  # Images that currently contain noise.
        timesteps: torch.Tensor,  # The current diffusion timestep for each image.
        predicted_noise: torch.Tensor,  # Noise predicted by the neural network.
    ) -> torch.Tensor:
        # Converts timestep values to integers and flattens them into a one-dimensional tensor.
        timesteps = timesteps.long().view(-1)

        # Retrieves the beta value for each image's current timestep.
        beta_t = self.betas[timesteps].view(-1, 1, 1, 1)

        # Retrieves the alpha value for each image's current timestep.
        alpha_t = self.alphas[timesteps].view(-1, 1, 1, 1)

        # Retrieves the cumulative alpha value for each image's current timestep.
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        # Estimates the average image at the previous timestep.
        # Subtracts the predicted noise contribution from the noisy image,
        # then divides by sqrt(alpha_t) to adjust for the noise schedule.
        mean = (noisy_images - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)

        # Creates a tensor filled with ones.
        # This represents the cumulative alpha value before the first timestep,
        # since there is no previous diffusion step at timestep zero.
        alpha_bar_prev = torch.ones_like(timesteps, dtype=self.alpha_bars.dtype, device=noisy_images.device)

        # Creates a Boolean mask identifying images that are not at timestep zero.
        nonzero_timesteps = timesteps > 0

        # For timesteps greater than zero, retrieves the cumulative alpha value
        # from the previous timestep.
        alpha_bar_prev[nonzero_timesteps] = self.alpha_bars[timesteps[nonzero_timesteps] - 1]

        # Reshapes the tensor so its values can broadcast across image dimensions.
        alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1, 1)

        # Calculates the posterior variance.
        # This determines how much additional random noise should be included
        # when estimating the previous diffusion step.
        posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

        # Prevents the variance from becoming too small or reaching zero,
        # which helps avoid numerical stability problems.
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        # Generates fresh random noise with the same dimensions as the noisy images.
        noise = torch.randn_like(noisy_images)

        # Creates a mask that is 1 for timesteps greater than zero and 0 at timestep zero.
        # This prevents additional random noise from being added at the final denoising step.
        noise_mask = nonzero_timesteps.float().view(-1, 1, 1, 1)

        # Produces the image at the previous timestep.
        # Adds scaled random noise to the calculated mean, except at timestep zero.
        return mean + torch.sqrt(posterior_variance) * noise * noise_mask
