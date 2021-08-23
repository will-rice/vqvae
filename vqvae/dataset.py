"""Dataset object to maintain train, validate, and test splits."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

from .config import Config


class Dataset:
    """Simple tfds wrapper."""

    def __init__(self, config: Config) -> None:
        self.config = config
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "speech_commands",
            split=["train", "validation", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        self._train = (
            ds_train.map(self.get_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self._test = (
            ds_test.map(self.get_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self._validate = (
            ds_val.map(self.get_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_spectrogram(self, audio, label):
        """Converts audio into a Mel-Spectrogram"""
        # Normalize
        audio = tf.cast(audio, tf.float32) / 32768.0
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        # Pad to same length
        audio = tf.cast(audio, tf.float32)
        equal_length = tf.concat([audio, zero_padding], 0)
        # Calculate spectrogram
        spectrogram = tfio.audio.spectrogram(
            equal_length,
            nfft=self.config.n_fft,
            window=self.config.window_length,
            stride=self.config.hop_length,
        )
        # Apply melscale to produce a melspectrogram
        mel_spectrogram = tfio.audio.melscale(
            spectrogram,
            rate=self.config.sample_rate,
            mels=self.config.n_mels,
            fmin=self.config.freq_min,
            fmax=self.config.freq_max,
        )
        # Convert to db
        mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
        return mel_spectrogram, label

    @property
    def train(self) -> tf.data.Dataset:
        """Train dataset."""
        return self._train

    @property
    def validate(self) -> tf.data.Dataset:
        """Validate dataset."""
        return self._validate

    @property
    def test(self) -> tf.data.Dataset:
        """Test dataset."""
        return self._test
