import matplotlib.pyplot as plt
import numpy as np


def plot_attack_comparison(original_images, adversarial_images, labels,
                           predictions, title="I-FGSM Attack Results"):
    """Show original vs adversarial images side by side with labels."""
    n = len(original_images)
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.subplot(n, 2, i * 2 + 1)
        plt.imshow(original_images[i].cpu().squeeze(), cmap="gray")
        plt.title(f"Orig: {labels[i].item()}")
        plt.axis("off")

        plt.subplot(n, 2, i * 2 + 2)
        plt.imshow(adversarial_images[i].detach().cpu().squeeze(), cmap="gray")
        plt.title(f"Adv: {predictions[i].item()}")
        plt.axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_defense_comparison(original, attacked, defended_dict, labels,
                            predictions_dict):
    """Show original, attacked, and multiple defense results per image."""
    n = len(original)
    num_cols = 2 + len(defended_dict)

    for i in range(n):
        plt.figure(figsize=(5 * num_cols, 4))

        plt.subplot(1, num_cols, 1)
        plt.imshow(original[i].cpu().squeeze(), cmap="gray")
        plt.title(f"Original ({labels[i].item()})")
        plt.axis("off")

        plt.subplot(1, num_cols, 2)
        plt.imshow(attacked[i].cpu().detach().squeeze(), cmap="gray")
        plt.title("Attacked")
        plt.axis("off")

        for idx, (name, defended_imgs) in enumerate(defended_dict.items()):
            plt.subplot(1, num_cols, idx + 3)
            plt.imshow(defended_imgs[i].cpu().detach().squeeze(), cmap="gray")
            pred = predictions_dict[name][i].item()
            plt.title(f"{name}\nPred: {pred}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def plot_defense_heatmap(results_dict, save_path=None):
    """Plot a seaborn heatmap of defense outcomes (Success/Partial/Failed)."""
    import seaborn as sns

    defense_names = list(results_dict.keys())
    outcomes = ["Success", "Partial", "Failed"]
    data = np.array([results_dict[name] for name in defense_names])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data, annot=True, fmt="d",
                xticklabels=outcomes,
                yticklabels=defense_names,
                cmap="RdYlGn", vmin=0, vmax=5)
    ax.set_title("Defense Performance Heatmap (out of 5 images)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_attack_grid(original_images, adversarial_images, perturbations=None,
                     labels=None, adv_predictions=None, save_path=None):
    """Show a grid of original, adversarial, and magnified perturbation images."""
    n = len(original_images)
    fig, axes = plt.subplots(n, 3, figsize=(8, 14))
    col_titles = ["Original", "Adversarial", "Perturbation (x10)"]

    for i in range(n):
        orig = original_images[i].cpu().squeeze()
        adv = adversarial_images[i].detach().cpu().squeeze()
        if perturbations is not None:
            diff = perturbations[i].detach().cpu().squeeze()
        else:
            diff = (adv - orig) * 10

        axes[i, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        if labels is not None:
            axes[i, 0].set_ylabel(f"True: {labels[i].item()}", fontsize=10)

        axes[i, 1].imshow(adv, cmap="gray", vmin=0, vmax=1)
        if adv_predictions is not None and i == 0:
            axes[i, 1].set_title(f"Pred: {adv_predictions[i].item()}")

        axes[i, 2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontweight="bold")

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("I-FGSM Attack Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
