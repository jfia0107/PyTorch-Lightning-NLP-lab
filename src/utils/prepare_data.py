"""
scripts/prepare_data.py

Standalone data preprocessing: download, tokenize/numericalize, split and save.
Two distinct "lanes" - encoder and CDL decide which pipeline is used.
Run this BEFORE training with all desired model/pipeline parameters set.
Everything is set in Hydra.
Usage:
    python -m src.utils.prepare_data.py
"""

import os
import json
import logging
from collections import Counter
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from tqdm import tqdm
import hydra
# PLUS spacy imported lower in code in case of CDL lane.
log = logging.getLogger(__name__)


def prepare_data(cfg: DictConfig):
    dc = cfg.data
    lane = dc.get("lane", "default")
    processed_path = f"./data/{dc.output_name}_{lane}_processed"

    if os.path.exists(processed_path):
        log.info(f"Processed data already exists at {processed_path}")
        return

    log.info(f"Preparing data: {processed_path}")

    # Load from source
    if dc.source_type == "hub":
        hf_dataset = load_dataset(dc.data_path, split="train")
    else:
        path = f"./data/{cfg.dataset_name}.{dc.data_format}"
        hf_dataset = load_dataset(dc.data_format, data_files=path, split="train")

    # Rename to desired column names
    hf_dataset = hf_dataset.rename_columns({
        dc.text_column: "text",
        dc.label_column: "label",
    })

    # Generate splits
    train_val = hf_dataset.train_test_split(test_size=dc.test_size, seed=42)
    val_proportion = dc.validation_size / (1.0 - dc.test_size)
    splits = train_val["train"].train_test_split(test_size=val_proportion, seed=42)

    dataset = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"],
        "test": train_val["test"],
    })

    # Encoder lane -> tokenize with HF tokenizer from select model
    tokenizer_name = _get_tokenizer_name(cfg)
    if tokenizer_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=dc.max_len,
            )

        dataset = dataset.map(tokenize_fn, batched=True, num_proc=os.cpu_count(), desc="Tokenizing")

    # Vocab/CDL lane -> build word2idx and numericalize (using spacy)
    elif dc.get("build_vocab"):
        import spacy
        word2idx = _build_vocab(dataset["train"]["text"], dc.max_vocab_size)

        vocab_path = dc.vocab_path
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(word2idx, f)
        log.info(f"Vocab saved to {vocab_path} ({len(word2idx)} tokens)")

        def numericalize_fn(examples):
            nlp = spacy.blank("en")
            results = []
            for doc in nlp.pipe(examples["text"], batch_size=1000, n_process=os.cpu_count()):
                tokens = [word2idx.get(t.text.lower(), word2idx["<UNK>"]) for t in doc]
                tokens = tokens[:dc.max_len]
                tokens += [word2idx["<PAD>"]] * max(0, dc.max_len - len(tokens))
                results.append(tokens)
            return {"input_ids": results}

        dataset = dataset.map(numericalize_fn, batched=True, num_proc=os.cpu_count(), desc="Numericalizing")

    # Save processed dataset
    dataset.save_to_disk(processed_path)
    log.info(f"Saved processed dataset to {processed_path}")

# Name of transformer tokenizer
def _get_tokenizer_name(cfg):
    model = cfg.get("model")
    if model is not None:
        backbone = model.get("backbone")
        if backbone is not None:
            return backbone.get("pretrained_model_name_or_path")
    return None

# build vocab for CDL lane
def _build_vocab(texts, max_vocab_size):
    import spacy
    nlp = spacy.blank("en")
    words = []
    for doc in tqdm(nlp.pipe(texts, batch_size=4000, n_process=os.cpu_count()), total=len(texts), desc="Building vocab"):
        words.extend([t.text.lower() for t in doc])
    counts = Counter(words)
    word2idx = {
        word: idx + 2
        for idx, (word, _) in enumerate(counts.most_common(max_vocab_size - 2))
    }
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    return word2idx

# hydra configs applied to main function
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    prepare_data(cfg)


if __name__ == "__main__":
    main()
