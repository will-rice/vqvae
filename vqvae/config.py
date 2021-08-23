"""Configuration class that holds hyperparameters."""
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Config:
    """Hyperparameters."""

    # training
    epochs: int = 30
    batch_size: int = 128
    steps_per_checkpoint: int = 1000
    val_steps: int = 100
    test_steps: int = 1

    # optimizer
    learning_rate: float = 1e-6
    beta_1 = 0.9
    beta_2 = 0.98
    epsilon = 1e-9
    decay_steps: int = 100000
    decay_rate: float = 0.0
    clip_norm: float = 1.0
    weight_decay_rate: float = 1e-6
    warmup_steps: int = 10000

    # dataset
    dataset: str = "speech_commands"

    # audio preprocessing
    sample_rate: int = 16000
    freq_min: int = 0
    freq_max: int = sample_rate // 2
    n_fft: int = 512
    window_length: int = int(0.025 * sample_rate)
    hop_length: int = int(0.010 * sample_rate)
    n_mels: int = 80
    max_spec_length: int = 1700

    # reproducibility
    seed: int = 1234

    def asdict(self) -> Dict[str, Any]:
        """Return Config object as dictionary."""
        return asdict(self)
