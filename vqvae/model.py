"""VQ-VAE Model."""
from typing import Any

import tensorflow as tf

from .layers import Decoder, Encoder, VectorQuantizer


class VQVAE(tf.keras.Model):
    """VQ-VAE Model."""

    def __init__(
        self,
        num_embeddings: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=latent_dim
        )
        self.encoder = Encoder(latent_dim=latent_dim, num_layers=num_layers)
        self.decoder = Decoder(num_layers=num_layers)

        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        """Foward Pass."""
        encoder_outputs = self.encoder(inputs, training=training)
        quantized_latents = self.vq_layer(encoder_outputs, training=training)
        reconstructions = self.decoder(quantized_latents, training=training)
        return reconstructions

    @property
    def metrics(self):
        """Model metrics."""
        return [self.total_loss, self.reconstruction_loss, self.vq_loss]

    def train_step(self, data):
        """Train step."""
        spec, label = data
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self(spec, training=True)

            # Calculate the losses.
            reconstruction_loss = tf.reduce_mean(
                (spec - reconstructions) ** 2
            ) / tf.math.reduce_variance(spec)
            total_loss = reconstruction_loss + sum(self.vq_layer.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_layer.losses))

        # Log results.
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        """Test step."""
        spec, label = data
        reconstructions = self(spec, training=True)

        # Calculate the losses.
        reconstruction_loss = tf.reduce_mean(
            (spec - reconstructions) ** 2
        ) / tf.math.reduce_variance(spec)
        total_loss = reconstruction_loss + sum(self.vq_layer.losses)

        # Update metrics
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_layer.losses))

        # Log results.
        return {metric.name: metric.result() for metric in self.metrics}
