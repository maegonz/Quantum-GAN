import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from qiskit_machine_learning.utils import algorithm_globals
from IPython.display import clear_output

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across torch and Qiskit.
    
    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    torch.manual_seed(seed)
    algorithm_globals.random_seed = seed

    return algorithm_globals.random_seed


# def plot_training_progress():
#     # we don't plot if we don't have enough data
#     if len(generator_loss_values) < 2:
#         return

#     clear_output(wait=True)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

#     # Generator Loss
#     ax1.set_title("Loss")
#     ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
#     ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
#     ax1.legend(loc="best")
#     ax1.set_xlabel("Iteration")
#     ax1.set_ylabel("Loss")
#     ax1.grid()

#     # Relative Entropy
#     ax2.set_title("Relative entropy")
#     ax2.plot(entropy_values)
#     ax2.set_xlabel("Iteration")
#     ax2.set_ylabel("Relative entropy")
#     ax2.grid()

#     plt.show()

def plot_generated_images(generator, n_qubits, num_images=16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, n_qubits)
        generated_images = generator(noise).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_images):
        ax = axes[i // 4, i % 4]
        ax.imshow(generated_images[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle("Generated Images")
    plt.show()

def total_variation_loss(image):
    # image shape: (batch, 1, 28, 28)
    tv_h = torch.pow(image[:,:,1:,:] - image[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(image[:,:,:,1:] - image[:,:,:,:-1], 2).sum()
    return (tv_h + tv_w)

def calculate_gradient_penalty(critic, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.InstanceNorm2d):
        # Only initialize if affine=True (weights/bias are not None)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)