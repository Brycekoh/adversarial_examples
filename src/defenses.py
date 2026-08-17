import io
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


def gaussian_blur_defense(images, kernel_size=5, sigma=1.5):
    """Apply Gaussian blur as a defense to a batch of images."""
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(images)


def jpeg_compression_defense(images, quality=15):
    """Apply JPEG compression as a defense, processing each image independently."""
    device = images.device
    compressed = []
    for img in images:
        pil_img = T.ToPILImage()(img.detach().cpu())
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed.append(T.ToTensor()(Image.open(buffer)))
    return torch.stack(compressed).to(device)


def quantization_defense(images, bits=3):
    """Quantize pixel values to a reduced number of levels."""
    levels = 2 ** bits - 1
    return torch.round(images * levels) / levels


def compute_signal_retention(current_step, schedule_length=1000,
                             singularity_offset=0.008):
    """Cosine-squared noise schedule from Improved DDPM (Nichol & Dhariwal, 2021)."""
    cosine_curve = lambda step: (
        torch.cos(
            torch.tensor(
                (step / schedule_length + singularity_offset)
                / (1 + singularity_offset)
                * (math.pi / 2)
            )
        ) ** 2
    )
    signal_retention = (cosine_curve(current_step) / cosine_curve(0)).item()
    return signal_retention


def adaptive_gaussian_denoise(noisy_image, noise_level):
    """Denoise using Gaussian blur with strength proportional to noise level."""
    blur_kernel_size = max(3, int(noise_level * 30) | 1)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    smoothed_image = gaussian_blur(
        noisy_image,
        kernel_size=[blur_kernel_size, blur_kernel_size],
        sigma=noise_level * 10
    )

    smoothing_strength = min(noise_level * 3, 0.9)
    denoised_image = (1 - smoothing_strength) * noisy_image \
                   + smoothing_strength * smoothed_image
    return torch.clamp(denoised_image, 0, 1)


def run_single_purification(adversarial_image, noise_fraction=0.15,
                            denoising_steps=10):
    """Run one forward-then-reverse diffusion purification pass."""
    SCHEDULE_LENGTH = 1000

    timestep_index = int(noise_fraction * SCHEDULE_LENGTH)
    signal_retention = compute_signal_retention(timestep_index, SCHEDULE_LENGTH)

    # Forward pass: corrupt with calibrated noise
    random_noise = torch.randn_like(adversarial_image)
    corrupted_image = (
        (signal_retention ** 0.5) * adversarial_image
        + ((1 - signal_retention) ** 0.5) * random_noise
    )

    # Reverse pass: iteratively denoise
    current_image = corrupted_image.clone()
    for denoising_step in range(denoising_steps, 0, -1):
        remaining_noise_fraction = denoising_step / denoising_steps
        current_noise_level = (
            (1 - signal_retention) ** 0.5
        ) * remaining_noise_fraction
        current_noise_level = max(current_noise_level, 0.01)
        current_image = adaptive_gaussian_denoise(current_image,
                                                  current_noise_level)

    return torch.clamp(current_image, 0, 1)


def purify_with_majority_vote(model, adversarial_images, device,
                              noise_fraction=0.25, denoising_steps=10,
                              num_runs=10):
    """Ensemble purification with majority vote across multiple denoising runs."""
    batch_size = adversarial_images.shape[0]
    num_classes = 10

    run_predictions = torch.zeros(num_runs, batch_size, dtype=torch.long)
    run_probabilities = torch.zeros(num_runs, batch_size, num_classes)
    run_purified_images = []

    model.eval()
    with torch.no_grad():
        for run_index in range(num_runs):
            purified_batch = run_single_purification(
                adversarial_images,
                noise_fraction=noise_fraction,
                denoising_steps=denoising_steps,
            )
            classifier_logits = model(purified_batch.to(device))
            class_probabilities = torch.softmax(classifier_logits, dim=1).cpu()
            run_predictions[run_index] = class_probabilities.argmax(dim=1)
            run_probabilities[run_index] = class_probabilities
            run_purified_images.append(purified_batch.cpu())

    # Majority voting
    final_predictions = torch.zeros(batch_size, dtype=torch.long)
    final_confidence = torch.zeros(batch_size)
    representative_images = []

    for image_index in range(batch_size):
        predictions_for_image = run_predictions[:, image_index]
        vote_counts = torch.bincount(predictions_for_image,
                                     minlength=num_classes)
        winning_class = vote_counts.argmax().item()
        final_predictions[image_index] = winning_class

        agreeing_runs = predictions_for_image == winning_class
        final_confidence[image_index] = (
            run_probabilities[agreeing_runs, image_index, winning_class].mean()
        )

        first_agreeing_run = agreeing_runs.nonzero(as_tuple=True)[0][0].item()
        representative_images.append(
            run_purified_images[first_agreeing_run][image_index]
        )

    representative_purified = torch.stack(representative_images)
    return final_predictions, final_confidence, representative_purified
