import torch
import torch.nn as nn


def ifgsm_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40,
                 device=None, clamp_min=0.0, clamp_max=1.0,
                 label_mapping=None):
    """Run an I-FGSM adversarial attack and return adversarial images and the original copies."""
    if device is None:
        device = images.device

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if label_mapping is not None:
        mapped_labels = torch.tensor(
            [label_mapping[l] for l in labels.cpu()],
            device=device
        )
    else:
        mapped_labels = labels

    loss_fn = nn.CrossEntropyLoss()
    adv_images = images.clone().detach().requires_grad_(True)

    for _ in range(iters):
        outputs = model(adv_images)
        loss = loss_fn(outputs, mapped_labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()

        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + eta, min=clamp_min, max=clamp_max)

        adv_images = adv_images.detach().requires_grad_(True)

    return adv_images.detach(), images.detach()
