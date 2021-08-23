"""Run model through training, validation, testing."""
import argparse
import logging
import os
from typing import Any

import tensorflow as tf

from vqvae.config import Config
from vqvae.dataset import Dataset
from vqvae.model import VQVAE
from vqvae.trainer import Trainer
from vqvae.utils import set_seeds

_LOGGER = logging.getLogger(__name__)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main() -> None:
    """Program Entry Point."""
    parser = argparse.ArgumentParser(description="Model Trainer")
    parser.add_argument("name", help="name of this training run")
    parser.add_argument(
        "--log_path", default="./logs", help="path to the tensorflow output logs"
    )
    args = parser.parse_known_args()[0]

    _train(args)


def _train(args: Any) -> None:
    log_path = os.path.join(args.log_path, args.name)
    os.makedirs(log_path, exist_ok=True)

    config = Config()
    set_seeds(config.seed)
    dataset = Dataset(config)
    model = VQVAE()

    _LOGGER.info("Initialize Trainer")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        log_path=log_path,
    )

    _LOGGER.info("Starting Training...")
    for epoch in range(config.epochs):
        # train
        trainer.train()
        # validate and checkpoint
        trainer.validate()
        # evaluate and score
        trainer.test()
        # reset metrics
        trainer.reset_states()


if __name__ == "__main__":
    main()
