"""
Embedding backbone for vocab-based models (BiLSTM, CNN).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def load_pretrained_vectors(path: str) -> dict:
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, vec_str = line.rstrip().split(" ", 1)
            try:
                vec = np.asarray(vec_str.split(), dtype="float32")
                if len(vec) == 300:
                    vectors[word] = vec
            except ValueError:
                continue
    return vectors


def create_embedding_backbone(
    vocab_path: str,
    embedding_dim: int,
    pretrained_file_path: Optional[str] = None,
    freeze: bool = False,
) -> nn.Embedding:
    # Creates the embedding layer
    with open(vocab_path, "r") as f:
        word2idx = json.load(f)

    vocab_size = len(word2idx)
    layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx["<PAD>"])
    # Initializes the embedding matrix with pre-trained word vectors and tracks vocabulary coverage
    if pretrained_file_path:
        vectors = load_pretrained_vectors(pretrained_file_path)
        matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
        hits, misses = 0, 0
        for word, idx in word2idx.items():
            vec = vectors.get(word)
            if vec is not None:
                matrix[idx] = vec
                hits += 1
            else:
                misses += 1
        print(f"Pretrained vectors: {hits} hits, {misses} misses")
        layer.load_state_dict({"weight": torch.from_numpy(matrix)})
    # in case embedding layer is frozen in Hydra
    if freeze:
        layer.weight.requires_grad = False

    return layer
