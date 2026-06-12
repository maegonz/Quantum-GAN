import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from .generator import Generator
from .discriminator import Discriminator
from .utils import total_variation_loss, calculate_gradient_penalty
from tqdm import tqdm


def training(generator: Generator,
             critic: Discriminator,
             train_loader: DataLoader,
             epochs: int,
             optimizer_G: Optimizer,
             optimizer_D: Optimizer,
             lambda_gp: float,
             lambda_tv: float,
             n_critic: int,
             device: torch.device):
    """
    Train the GAN for a specified number of epochs.
    """

    generator.to(device)
    critic.to(device)

    G_loss, D_loss = [], []

    epochs_tqdm = tqdm(range(epochs), desc="Training progress")

    for epoch in epochs_tqdm:
        for i, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Critic
            for _ in range(n_critic):
                optimizer_D.zero_grad()

                # Generate noise and fake images
                with torch.no_grad():
                    z_noise = torch.randn(batch_size, generator.n_qubits + generator.n_ancillas, device=device)
                    fake_imgs = generator(z_noise).float()  # Ensure fake images are float for loss calculation

                # Compute critic scores for real and fake images
                real_preds = critic(real_imgs)
                fake_preds = critic(fake_imgs)

                # Compute loss out of critic scores
                loss_real = torch.mean(real_preds)
                loss_fake = torch.mean(fake_preds)
                # Compute gradient penalty
                gp = calculate_gradient_penalty(critic, real_imgs, fake_imgs, device)
                # Compute total critic loss: WGAN-GP loss + gradient penalty
                d_loss = (loss_fake - loss_real) + lambda_gp * gp

                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            z_noise = torch.randn(batch_size, generator.n_qubits + generator.n_ancillas, device=device)
            fake_imgs = generator(z_noise)
            fake_preds = critic(fake_imgs)
            g_loss = -torch.mean(fake_preds) + lambda_tv * total_variation_loss(fake_imgs)

            g_loss.backward()
            optimizer_G.step()

            G_loss.append(g_loss.item())
            D_loss.append(d_loss.item())
            
        epochs_tqdm.set_postfix({"G Loss": g_loss.item(), "D Loss": d_loss.item()})

        if epoch > 0 and epoch % 2 == 0:
            torch.save(generator.state_dict(), f"./save/generator_epoch_{epoch}.pth")
            torch.save(critic.state_dict(), f"./save/critic_epoch_{epoch}.pth")
        
    return G_loss, D_loss