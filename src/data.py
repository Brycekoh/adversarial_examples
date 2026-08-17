import json
import os
import tarfile
import urllib.request

import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


IMAGENETTE_LABEL_MAP = {0: 0, 1: 217, 2: 482, 3: 491, 4: 497}
"""Maps Imagenette class indices to ImageNet-1K class indices (first 5 classes)."""


def load_mnist(batch_size_train=64, batch_size_test=1000):
    """Load MNIST train and test DataLoaders with ToTensor transform."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root="./data", train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size_train,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size_test,
                                              shuffle=False)
    return train_loader, test_loader


def load_imagenette(data_dir=".", num_samples=5):
    """Download Imagenette if needed, return images and labels for num_samples classes."""
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    imagenette_path = os.path.join(data_dir, "imagenette2")

    if not os.path.exists(imagenette_path):
        print("Downloading ImageNette...")
        tgz_path = os.path.join(data_dir, "imagenette2.tgz")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as f:
            f.extractall(data_dir, filter="data")
        os.remove(tgz_path)
        print("Done.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_set = ImageFolder(os.path.join(imagenette_path, "val"),
                           transform=transform)

    # Select one image per class
    class_indices = {i: None for i in range(num_samples)}
    for idx, (img, label) in enumerate(test_set):
        if label < num_samples and class_indices[label] is None:
            class_indices[label] = idx
        if all(v is not None for v in class_indices.values()):
            break

    images, labels = [], []
    for c in range(num_samples):
        img, label = test_set[class_indices[c]]
        images.append(img)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)
