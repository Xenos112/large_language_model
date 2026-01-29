import os


class Paths:
    """
    Centralized path management for the project.
        - DATA_DIR: root directory for all data
        - RAW_DATA_DIR: directory for raw data
        - PROCESSED_DATA_DIR: directory for processed data
        - TOKENIZER_FILE: path to tokenizer file
    """

    DATA_DIR = "./data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    TOKENIZER_FILE = os.path.join("./src/tokenization/tokenizer.json")


class SaveData:
    """
    Data saving configuration.
        - shard_size: size of each shard in bytes
    """

    shard_size = 1024 * 1024 * 1024  # 1GB per shard


class ModelConfig:
    """
    Model architecture configuration.
        - vocab_size: size of the vocabulary
        - context_length: maximum length of input sequence
        - embedding_dim: dimension of the token embeddings
        - num_heads: number of attention heads
        - num_layers: number of transformer layers
    """

    vocab_size = 32000
    context_length = 2048
    embedding_dim = 768
    num_heads = 12
    num_layers = 12
