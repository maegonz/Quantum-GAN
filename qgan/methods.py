import jax
import jax.numpy as jnp
import numpy as np
import optax

# @jax.jit(static_argnames=['model'])
# def loss_fn(model, weights, noise, target):
#     """
#     Compute the MSE loss between predictions and target.
    
#     Params
#     ------
#     predictions: jnp.ndarray
#         The predicted values from the generator.
#     target: jnp.ndarray
#         The real data we want to match.
#     """

#     # Get predictions from the model
#     predictions = model(weights, noise)
#     predictions = jnp.array(predictions)  # Convert back to JAX array for loss computation
#     mse  = jnp.mean((predictions - target) ** 2)
#     return mse

# @jax.jit(static_argnames=['model', 'optimizer'])
# def update(model, weights, noise, target, opt_state, optimizer):
#     def loss_fn_wrapper(weights):
#         return loss_fn(model, weights, noise, target)
#     # Compute loss and gradients
#     loss, grads = jax.value_and_grad(loss_fn_wrapper)(weights)

#     # Update model parameters
#     updates, opt_state = optimizer.update(grads, opt_state)
#     weights = optax.apply_updates(weights, updates)

#     return weights, opt_state, loss


def calculate_gradient_penalty(critic_fn, real_samples, fake_samples, rng_key):
    batch_size = real_samples.shape[0]
    
    # 1. Random weight term for interpolation
    alpha = jax.random.uniform(rng_key, shape=(batch_size, 1, 1, 1))
    
    # 2. Get random interpolation between real and fake
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    
    # 3. Define a wrapper that sums the critic outputs over the batch.
    # Differentiating a sum gives the per-sample gradients we need.
    def critic_sum(x):
        return jnp.sum(critic_fn(x))
        
    # 4. Get gradients of the critic_sum w.r.t. the interpolates
    gradients = jax.grad(critic_sum)(interpolates)
    
    # 5. Flatten the gradients for norm calculation
    gradients = gradients.reshape((batch_size, -1))
    
    # CRITICAL JAX TIP: Add a tiny epsilon (1e-12) before the sqrt! 
    # If gradients are perfectly 0, JAX's jnp.sqrt will return NaN during backprop.
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(gradients), axis=1) + 1e-12)
    
    # 6. Calculate the final penalty
    gradient_penalty = jnp.mean((l2_norm - 1.0) ** 2)
    
    return gradient_penalty


# ###########################################################################################
# ==================== --- Generator Loss --- ===========================================
# ###########################################################################################
def total_variation_loss(image):
    """
    Expects Flax native NHWC format: (batch, height, width, channels)
    e.g., (batch, 28, 28, 1)
    """
    # tv_h: differences along the Height axis (index 1)
    tv_h = jnp.sum(jnp.square(image[:, 1:, :, :] - image[:, :-1, :, :]))
    # tv_w: differences along the Width axis (index 2)
    tv_w = jnp.sum(jnp.square(image[:, :, 1:, :] - image[:, :, :-1, :]))
    
    return tv_h + tv_w

def gen_loss_fn(circuit, discriminator, disc_params, gen_weights, noise, lambda_tv):
    #  Generate fake image from quantum circuit
    fake_patch = circuit(noise, gen_weights)
    #  (Flax requires passing params explicitly)
    pred = discriminator.apply({'params': disc_params}, fake_patch)

    #  TV Loss
    tv_loss = total_variation_loss(fake_patch)

    #  Compute critic output
    return -jnp.mean(pred) + lambda_tv * tv_loss

@jax.jit(static_argnames=['circuit', 'discriminator'])
def update_generator(circuit, discriminator, disc_params, gen_weights, opt_state_G, optimizer_G, noise, lambda_tv=0.1):
    def gen_loss_fn_wrapper(gen_weights):
        return gen_loss_fn(circuit, discriminator, disc_params, gen_weights, noise, lambda_tv)
    
    # Differentiate ONLY with respect to gen_weights (argnums=0 is default)
    loss_G, grads_G = jax.value_and_grad(gen_loss_fn_wrapper)(gen_weights)

    updates_G, opt_state_G = optimizer_G.update(grads_G, opt_state_G)
    gen_weights = optax.apply_updates(gen_weights, updates_G)
    
    return gen_weights, opt_state_G, loss_G


# ###########################################################################################
# ==================== --- Discriminator Loss --- ===========================================
# ###########################################################################################
# The Discriminator wants to correctly identify Real (1) and Fake (0)
def disc_loss_fn(circuit, discriminator, disc_params, gen_weights, noise, real_patch):
    # 1. Get both patches
    fake_patch = circuit(noise, gen_weights)
    
    # 2. Predict on both
    real_pred = discriminator.apply({'params': disc_params}, real_patch)
    fake_pred = discriminator.apply({'params': disc_params}, fake_patch)
    
    # 3. Calculate Binary Cross Entropy
    loss_real = -jnp.mean(jnp.log(real_pred + 1e-8))
    loss_fake = -jnp.mean(jnp.log(1.0 - fake_pred + 1e-8))
    
    return loss_real + loss_fake

@jax.jit(static_argnames=['circuit', 'discriminator'])
def update_discriminator(circuit, discriminator, disc_params, gen_weights, opt_state_D, optimizer_D, noise, real_patch):
    def disc_loss_fn_wrapper(disc_params):
        return disc_loss_fn(circuit, discriminator, disc_params, gen_weights, noise, real_patch)
    
    # Differentiate only with respect to disc_params (argnums=0 is default)
    loss_D, grads_D = jax.value_and_grad(disc_loss_fn_wrapper)(disc_params)

    updates_D, opt_state_D = optimizer_D.update(grads_D, opt_state_D)
    disc_params = optax.apply_updates(disc_params, updates_D)
    
    return disc_params, opt_state_D, loss_D
