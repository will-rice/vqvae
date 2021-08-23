"""General purpose utilities."""
import random

import numpy as np
import tensorflow as tf


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
