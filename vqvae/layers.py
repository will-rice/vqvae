"""VQ-VAE Layers."""
from typing import Any

import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):
    """Vector quantizer layer."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer."""
        self.embeddings = self.add_weight(
            shape=(self.embedding_dim, self.num_embeddings),
            initializer="random_uniform",
            trainable=True,
            name="embeddings_vqvae",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        # Calculate the input shape of the inputs
        input_shape = tf.shape(inputs)
        # flatten while keeping embedding_dim
        flattened = tf.reshape(inputs, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - inputs) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized

    def get_code_indices(self, flattened_inputs: tf.Tensor) -> tf.Tensor:
        """Calculate L2-normalized distance between the inputs and the codebook."""
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class Encoder(tf.keras.layers.Layer):
    """VQ-VAE Encoder."""

    def __init__(
        self, latent_dim: int = 16, num_layers: int = 2, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.conv_layers = [
            tf.keras.layers.Conv1D(
                filters=32 if i == 0 else 32 * i,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
            )
            for i in range(num_layers)
        ]
        self.conv_final = tf.keras.layers.Conv1D(
            filters=latent_dim,
            kernel_size=1,
            padding="same",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        out = inputs
        for layer in self.conv_layers:
            out = layer(out)
        out = self.conv_final(out)
        return out


class Decoder(tf.keras.layers.Layer):
    """VQ-VAE Decoder."""

    def __init__(self, num_layers: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv_layers = [
            tf.keras.layers.Conv1DTranspose(
                filters=32 if i == 0 else 32 * i,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
            )
            for i in reversed(range(num_layers))
        ]
        self.conv_final = tf.keras.layers.Conv1DTranspose(
            filters=1, kernel_size=3, padding="same"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        out = inputs
        for layer in self.conv_layers:
            out = layer(out)
        out = self.conv_final(out)
        return out
