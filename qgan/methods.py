import jax
import jax.numpy as jnp
import numpy as np
import optax
import math
from functools import partial
from qgan.generator import Generator
from qgan.discriminator import Discriminator


# ###########################################################################################
# ==================== --- Generator Loss --- ===========================================
# ###########################################################################################
def total_variation_loss(image):
    """
    Expects Flax native NHWC format: (batch, height, width, channels)
    """
    # tv_h: differences along the Height axis (index 1)
    tv_h = jnp.sum(jnp.square(image[:, 1:, :, :] - image[:, :-1, :, :]))
    # tv_w: differences along the Width axis (index 2)
    tv_w = jnp.sum(jnp.square(image[:, :, 1:, :] - image[:, :, :-1, :]))
    
    return tv_h + tv_w

@jax.jit(static_argnames=['circuit', 'generator', 'discriminator', 'optimizer_G'])
def update_generator(circuit, 
                     generator: Generator, 
                     discriminator: Discriminator, 
                     disc_params, 
                     gen_weights: jnp.ndarray, 
                     opt_state_G: optax.OptState, 
                     optimizer_G, 
                     z_noise: jnp.ndarray, 
                     rng_key: jax.random.PRNGKey, 
                     lambda_tv: float=0.1):
    """
    Updates the generator's weights using the computed gradients.

    Parameters
    ----------
    circuit: function
        The quantum circuit function to generate patches.
    generator: Generator
        The Flax module for the generator.
    discriminator: Discriminator
        The Flax module for the discriminator.
    disc_params: dict
        The current parameters of the discriminator.
    gen_weights: jnp.ndarray
        The current trainable weights for the generator's quantum circuit.
    opt_state_G: optax.OptState
        The current optimizer state for the generator.
    optimizer_G: optax.Optimizer
        The Optax optimizer for the generator.
    z_noise: jnp.ndarray
        The input noise vector for the generator, shape (batch_size, n_qubits).
    rng_key: jax.random.PRNGKey
        The random key for generating random numbers.
    lambda_tv: float
        The weight for the Total Variation loss term in the generator's loss function.  Defaults to 0.1.
    
    Returns
    -------
    gen_weights: jnp.ndarray
        The updated trainable weights for the generator's quantum circuit after one optimization step.
    opt_state_G: optax.OptState
        The updated optimizer state for the generator.
    loss_G: float
        The computed loss for the generator before the update.
    """

    def gen_loss_fn(gen_weights):
        #  Generate fake image from generator
        fake_imgs = generator.apply({}, circuit, z_noise, gen_weights)

        #  Compute critic scores for fake images
        pred = discriminator.apply({'params': disc_params}, fake_imgs, rngs={'dropout': rng_key})

        #  TV Loss
        tv_loss = total_variation_loss(fake_imgs)

        #  Compute total generator loss: WGAN loss + TV regularization
        total_loss = -jnp.mean(pred) + lambda_tv * tv_loss
        return total_loss
    
    # Differentiate ONLY with respect to gen_weights (argnums=0 is default)
    loss_G, grads_G = jax.value_and_grad(gen_loss_fn)(gen_weights)

    updates_G, opt_state_G = optimizer_G.update(grads_G, opt_state_G)
    gen_weights = optax.apply_updates(gen_weights, updates_G)
    
    return gen_weights, opt_state_G, loss_G


# ###########################################################################################
# ==================== --- Discriminator Loss --- ===========================================
# ###########################################################################################
# The Discriminator wants to correctly identify Real (1) and Fake (0)
def calculate_gradient_penalty(critic_fn, real_samples, fake_samples, rng_key):
    batch_size = real_samples.shape[0]
    
    # Get Random weight term for interpolation
    alpha = jax.random.uniform(rng_key, shape=(batch_size, 1, 1, 1))
    
    # Get random interpolation between real and fake
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    
    # Define a wrapper that sums the critic outputs over the batch.
    # Differentiating a sum gives the per-sample gradients we need.
    def critic_sum(x):
        return jnp.sum(critic_fn(x))
        
    # Get gradients of the critic_sum w.r.t. the interpolates
    gradients = jax.grad(critic_sum)(interpolates)
    
    # Flatten the gradients for norm calculation
    gradients = gradients.reshape((batch_size, -1))
    
    # Add a tiny epsilon in order to avoid NaN during backprop
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(gradients), axis=1) + 1e-12)
    
    gradient_penalty = jnp.mean((l2_norm - 1.0) ** 2)
    
    return gradient_penalty

@jax.jit(static_argnames=['circuit', 'generator', 'discriminator', 'optimizer_D'])
def update_discriminator(circuit, 
                         generator: Generator, 
                         discriminator: Discriminator, 
                         disc_params, 
                         gen_weights: jnp.ndarray, 
                         opt_state_D: optax.OptState, 
                         optimizer_D, 
                         z_noise: jnp.ndarray, 
                         real_imgs: jnp.ndarray, 
                         lambda_gp: float, 
                         rng_key: jax.random.PRNGKey):
    """    
    Updates the discriminator's parameters using the computed gradients.

    Parameters
    ----------
    circuit: function
        The quantum circuit function to generate patches.
    generator: Generator
        The Flax module for the generator.
    discriminator: Discriminator
        The Flax module for the discriminator.
    disc_params: dict
        The current parameters of the discriminator.
    gen_weights: jnp.ndarray
        The current trainable weights for the generator's quantum circuit.
    opt_state_D: optax.OptState
        The current optimizer state for the discriminator.
    optimizer_D: optax.Optimizer
        The Optax optimizer for the discriminator.
    z_noise: jnp.ndarray
        The input noise vector for the generator, shape (batch_size, n_qubits).
    real_imgs: jnp.ndarray
        The batch of real images, shape (batch_size, 28, 28, 1).
    lambda_gp: float
        The weight for the gradient penalty term in the WGAN-GP loss.
    rng_key: jax.random.PRNGKey
        The random key for generating random numbers.

    Returns
    -------
    disc_params: dict
        The updated parameters of the discriminator after one optimization step.
    opt_state_D: optax.OptState
        The updated optimizer state for the discriminator.
    loss_D: float
        The computed loss for the discriminator before the update.
    """
    key_real, key_fake, key_gp = jax.random.split(rng_key, 3)

    def disc_loss_fn(params):
        # Generate fake images
        fake_imgs = generator.apply({}, circuit, z_noise, gen_weights)

        # Compute critic scores for real and fake images
        real_pred = discriminator.apply({'params': params}, real_imgs, rngs={'dropout': key_real})
        fake_pred = discriminator.apply({'params': params}, fake_imgs, rngs={'dropout': key_fake})
        
        # Calculate loss out of critic scores
        loss_real = jnp.mean(real_pred)
        loss_fake = jnp.mean(fake_pred)

        # Gradient the penalty
        critic_fn = lambda x: discriminator.apply({'params': params}, x, training=False)
        gp = calculate_gradient_penalty(critic_fn, real_imgs, fake_imgs, key_gp)

        # WGAN-GP Loss: (fake_validity - real_validity) + λ * gp
        total_loss = (loss_fake - loss_real) + lambda_gp * gp
        return total_loss
    
    # Differentiate only with respect to disc_params (argnums=0 is default)
    loss_D, grads_D = jax.value_and_grad(disc_loss_fn)(disc_params)

    updates_D, opt_state_D = optimizer_D.update(grads_D, opt_state_D)
    disc_params = optax.apply_updates(disc_params, updates_D)
    
    return disc_params, opt_state_D, loss_D
