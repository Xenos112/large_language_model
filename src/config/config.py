
import os


class Paths:
    """Centralized path management for the project."""
    DATA_DIR = "./data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    TOKENIZER_FILE = "./src/tokenization/tokenizer.json"  # Fixed: tokenizer not tokenizing


class SaveData:
    """Data saving configuration."""
    shard_size = 1024 * 1024 * 1024  # 1GB per shard


class ModelConfig:
    """Model architecture configuration."""
    vocab_size = 32000  # Standard LLaMA vocab size as starting point
    context_length = 2048  # Add context length for training
    embedding_dim = 768   # Add embedding dimension
    num_heads = 12        # Attention heads
    num_layers = 12       # Transformer layers
