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
        - hidden_dim: dimension of the hidden state
        - epsilon: small value to prevent division by zero
        - max_sequence_length: maximum sequence length for input sequences
        - base: base value for positional encoding
    """

    vocab_size = 32000
    hidden_dim = 1024
    epsilon = 1e-8
    max_sequence_length = 4096
    base = 10000.0
