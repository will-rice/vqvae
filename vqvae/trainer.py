"""Simple Model Trainer."""
import logging

import tensorflow as tf
import tensorflow_addons as tfa
from wandb.keras import WandbCallback

from vqvae.config import Config
from vqvae.dataset import Dataset
from vqvae.summary import Summary

_LOGGER = logging.getLogger(__name__)


class Trainer:
    """Simple Trainer."""

    def __init__(
        self, config: Config, model: tf.keras.Model, dataset: Dataset, log_path: str
    ) -> None:
        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_path = log_path
        self.summary = Summary(config, log_path)
        self.callbacks = (
            [
                WandbCallback(),
            ],
        )
        self._setup()
        self._restore()

    def _setup(self) -> None:

        self._optimizer = tfa.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay_rate,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            clipnorm=self.config.clip_norm,
        )

        self.model.compile(
            optimizer=self._optimizer,
        )

    def _restore(self) -> None:
        """Restore From Checkpoint."""
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, tf.int64),
            model=self.model,
            optimizer=self._optimizer,
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.log_path, max_to_keep=5
        )

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            _LOGGER.info("Latest checkpoint restored!!")

    def train(self) -> None:
        """Train loop."""
        self.model.fit(self.dataset.train, callbacks=self.callbacks)

    def validate(self) -> None:
        """Validate loop."""
        result = self.model.evaluate(self.dataset.validate, return_dict=True)
        self.summary.write_metrics({"validate": result})

    def test(self) -> None:
        """Test Loop."""
        result = self.model.evaluate(self.dataset.test, return_dict=True)
        self.summary.write_metrics({"test": result})

    def reset_states(self) -> None:
        """Reset metrics."""
        for metric in self.model.metrics:
            metric.reset_states()

    @property
    def step(self) -> int:
        """Global step."""
        return self.ckpt.step.numpy()
