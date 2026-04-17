"""
Custom data samplers go here.
Usage:
    Selected and configured in Hydra configs.
"""

import logging
import numpy as np
from torch.utils.data import Sampler

log = logging.getLogger(__name__)

class StratifiedOversampleSampler(Sampler):
    """
    Stratified oversampling: ensures each batch has a controlled minority ratio.
    Oversamples minority.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, minority_ratio: float = 0.5):
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.minority_ratio = minority_ratio
        # thresholds were set specifically for a select dataset, user might want to change this according to needs.
        self.majority_idx = np.where(self.labels < 0.5)[0]
        self.minority_idx = np.where(self.labels >= 0.5)[0]

        self.minority_count = int(self.batch_size * self.minority_ratio)
        self.majority_count = self.batch_size - self.minority_count
        self.num_batches = len(self.majority_idx) // max(self.majority_count, 1)

        log.info(
            f"StratifiedSampler batches={self.num_batches} samples_per_batch={self.majority_count}maj+{self.minority_count}min")

    def __iter__(self):
        majority_shuffled = np.random.permutation(self.majority_idx)
        total_minority_needed = self.num_batches * self.minority_count
        minority_sampled = np.random.choice(self.minority_idx, size=total_minority_needed, replace=True)

        all_indices = []
        for i in range(self.num_batches):
            maj = majority_shuffled[i * self.majority_count:(i + 1) * self.majority_count]
            mn = minority_sampled[i * self.minority_count:(i + 1) * self.minority_count]
            batch = np.concatenate([maj, mn])
            np.random.shuffle(batch)
            all_indices.extend(batch.tolist())

        return iter(all_indices)

    def __len__(self):
        return self.num_batches * self.batch_size