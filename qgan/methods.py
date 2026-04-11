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

@jax.jit(static_argnames=['circuit', 'generator', 'discriminator'])
def update_generator(circuit, generator, discriminator, disc_params, gen_weights, opt_state_G, optimizer_G, z_noise, lambda_tv=0.1):
    def gen_loss_fn(gen_weights):
        #  Generate fake image from generator
        fake_imgs = generator(circuit, z_noise, gen_weights)

        #  Compute critic scores for fake images
        pred = discriminator.apply({'params': disc_params}, fake_imgs)

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

@jax.jit(static_argnames=['circuit', 'generator', 'discriminator'])
def update_discriminator(circuit, generator, discriminator, disc_params, gen_weights, opt_state_D, optimizer_D, z_noise, real_imgs, lambda_gp, rng_key):
    def disc_loss_fn(params):
        # Generate fake images
        fake_imgs = generator(circuit, z_noise, gen_weights)
        
        # Compute critic scores for real and fake images
        real_pred = discriminator.apply({'params': params}, real_imgs)
        fake_pred = discriminator.apply({'params': params}, fake_imgs)
        
        # Calculate loss out of critic scores
        loss_real = -jnp.mean(real_pred)
        loss_fake = -jnp.mean(fake_pred)

        # Gradient the penalty
        critic_fn = lambda x: discriminator.apply({'params': params}, x)
        gp = calculate_gradient_penalty(critic_fn, real_imgs, fake_imgs, rng_key)

        # WGAN-GP Loss: (fake_validity - real_validity) + λ * gp
        total_loss = (loss_fake - loss_real) + lambda_gp * gp
        return total_loss
    
    # Differentiate only with respect to disc_params (argnums=0 is default)
    loss_D, grads_D = jax.value_and_grad(disc_loss_fn)(disc_params)

    updates_D, opt_state_D = optimizer_D.update(grads_D, opt_state_D)
    disc_params = optax.apply_updates(disc_params, updates_D)
    
    return disc_params, opt_state_D, loss_D
