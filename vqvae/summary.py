"""Summary writer for wandb/tensorboard."""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from .config import Config


class Summary:
    """Summary writer."""

    def __init__(self, config: Config, log_path: str) -> None:
        self.config = config
        self.log_path = log_path
        self.trial_id = os.path.split(log_path)[1]
        wandb.init(
            project=f"vqvae",
            id=self.trial_id,
            name=self.trial_id,
            dir=log_path,
            config=config.asdict(),
        )

    def write_audio(self, audio: np.ndarray, step: int) -> None:
        """Audio writer."""
        wandb.log(
            {"audio": wandb.Audio(audio, sample_rate=self.config.sample_rate)},
            step=step,
        )

    @staticmethod
    def write_images(images, step) -> None:
        """Image writer."""
        wandb.log(
            {"images": wandb.Image(images)},
            step=step,
        )

    @staticmethod
    def write_metrics(metrics) -> None:
        """Metric writer."""
        wandb.log(metrics)
