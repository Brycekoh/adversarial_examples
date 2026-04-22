# Adversarial Robustness: I-FGSM Attacks and Diffusion-Based Purification

## Project Summary
This project evaluates the security of Deep Neural Networks (DNNs) against gradient-based adversarial perturbations. By implementing the Iterative Fast Gradient Sign Method (I-FGSM), the research demonstrates how minimal, targeted noise can bypass state-of-the-art classification models. To mitigate these risks, a Stochastic Purification pipeline was developed, utilizing an adaptive denoising process to restore model accuracy without requiring architectural retraining or fine-tuning of the base classifier.

## Technical Implementation

### 1. Adversarial Attack Methodology
The primary threat model utilized is the Iterative Fast Gradient Sign Method (I-FGSM). Unlike single-step attacks, I-FGSM applies perturbations iteratively to ensure the adversarial example remains within a defined epsilon-ball while maximizing the loss function across multiple steps.

**Technical Specifications:**
* **Optimization:** Directed gradient descent on input pixel space.
* **Architectures Tested:** ResNet50 (Pre-trained on ImageNet-1K) and custom 4-layer Convolutional Neural Networks (MNIST).
* **Constraints:** Implementation of L-infinity epsilon clipping and sign-based normalization to maintain image integrity while maximizing misclassification.

### 2. Purification and Defense Strategy
The defense architecture is inspired by the forward and reverse processes of Diffusion Models. The underlying theory assumes that adversarial perturbations exist as structured, high-frequency signals that can be effectively disrupted through controlled stochastic noise injection.

**Defense Pipeline:**
* **Forward Noise Injection:** Application of Gaussian noise according to a cosine-squared schedule to neutralize adversarial gradients.
* **Adaptive Denoising:** Implementation of an adaptive Gaussian filter where the kernel size and standard deviation are dynamically calculated based on the noise-to-signal ratio of the specific diffusion timestep.
* **Comparative Baseline:** Evaluated against traditional defenses including JPEG compression (Quality = 15) and 3-bit quantization to benchmark the efficacy of stochastic purification.

## Experimental Results

### ResNet50 Performance on ImageNette (10-Class Subset)
| Category | Original Prediction | Attacked Prediction | Confidence |
| :--- | :--- | :--- | :--- |
| Image 0 | Tench | Reel | 100.0% |
| Image 1 | English Springer Spaniel | Shih Tzu | 100.0% |
| Image 3 | Chainsaw | Lorikeet | 100.0% |

### Defense Efficacy Analysis (MNIST)
* **Baseline (No Defense):** 0% Accuracy under I-FGSM (eps=0.3).
* **Standard Gaussian Blur:** 40% Accuracy restoration.
* **Diffusion Purification (Adaptive):** 60% Accuracy restoration with 80% total attack disruption.

## Environment and Dependencies
The project is implemented in Python using the PyTorch framework. Hardware acceleration via CUDA is utilized for the iterative denoising passes and high-dimensional gradient calculations.

**Core Libraries:**
* torch
* torchvision
* matplotlib
* numpy
* PIL (Pillow)

## Project Structure
* **Attack Module:** Logic for generating adversarial perturbations using I-FGSM.
* **Purification Logic:** Implementation of the cosine noise schedule and the adaptive denoising loop.
* **Evaluation Suite:** Scripts for measuring classification success rates and confidence intervals across various defense thresholds.

## Key Outcomes
* **Robustness Identification:** Quantified the epsilon thresholds where traditional data-transformation defenses fail against iterative attacks.
* **Stochastic Recovery:** Demonstrated that non-targeted noise injection followed by structural denoising is significantly more effective than direct filtering.
* **Operational Optimization:** Utilized bitwise operations for dynamic kernel sizing to minimize computational overhead during the purification process, ensuring scalability for real-time inference environments.

---
