"""
Code for tokenizing text data using HuggingFace tokenizers library.
Implements efficient ByteLevel BPE tokenization with streaming from processed shards.
"""

from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.config.config import ModelConfig, Paths
from src.utils.logger import Logger


def shard_iterator() -> Iterator[str]:
    """
    Yield text lines from all processed shards.

    Yields:
        str: Individual article texts
    """
    logger = Logger(path="tokenizing.shard_iterator")
    shards = sorted(Path(Paths.PROCESSED_DATA_DIR).glob("*.txt"))

    if not shards:
        logger.log("No shards found! Run process_data.py first", level="ERROR")
        raise FileNotFoundError("No processed shards found")

    logger.log(f"Found {len(shards)} shards to tokenize")

    for shard in shards:
        logger.log(f"Processing {shard.name}")
        try:
            with open(shard, "r", encoding="utf-8") as file:
                # Split by double newlines to get individual articles
                content = file.read()
                articles = content.split("\n\n")

                for line in articles:
                    if line.strip():  # Skip empty lines
                        yield line
        except Exception as e:
            logger.log(f"Error reading {shard}: {e}", level="WARNING")
            continue


def build_tokenizer() -> Tokenizer:
    """
    Build a new ByteLevel BPE tokenizer with optimized settings.

    Returns:
        Tokenizer: A configured tokenizer ready for training
    """
    # Define special tokens with <|...|> syntax
    special_tokens = [
        "<|EOS|>",
        "<|PAD|>",
        "<|UNK|>",
        "<|BOS|>",
    ]

    # Create BPE model with ByteLevel encoding
    tokenizer = Tokenizer(BPE())

    # Configure normalizer: lowercase + remove accents for efficiency
    tokenizer.normalizer = Sequence(
        [
            NFD(),
            Lowercase(),
            StripAccents(),
        ]
    )

    # Use ByteLevel pre-tokenizer (most efficient for LLMs)
    tokenizer.pre_tokenizer = ByteLevel()

    # Configure decoder to reconstruct text from ByteLevel tokens
    tokenizer.decoder = ByteLevelDecoder()

    # Create trainer with specified vocabulary size
    trainer = BpeTrainer(
        vocab_size=ModelConfig.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Ignore very rare tokens
        show_progress=True,
    )

    # Store trainer for use in training function
    tokenizer.trainer = trainer

    return tokenizer


def train_tokenizer(tokenizer: Tokenizer) -> None:
    """
    Train tokenizer on text from processed shards.

    Args:
        tokenizer: The tokenizer instance to train
    """
    logger = Logger(path="tokenizing.train_tokenizer")
    logger.log(f"Training with vocab_size={ModelConfig.vocab_size}")

    # Train from iterator (memory efficient - processes one line at a time)
    tokenizer.train_from_iterator(
        shard_iterator(),
        trainer=tokenizer.trainer,
    )

    logger.log("Training complete!", level="SUCCESS")


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """
    Save tokenizer in native HuggingFace format.

    Args:
        tokenizer: The trained tokenizer to save
        path: File path where to save the tokenizer
    """
    logger = Logger(path="tokenizing.save_tokenizer")

    # Create parent directories if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save in native HuggingFace JSON format
    tokenizer.save(path)
    logger.log(f"Tokenizer saved to {path}", level="SUCCESS")


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a saved tokenizer from file.

    Args:
        path: File path to the saved tokenizer

    Returns:
        Tokenizer: The loaded tokenizer
    """
    logger = Logger(path="tokenizing.load_tokenizer")

    if not Path(path).exists():
        logger.log(f"Tokenizer file not found: {path}", level="ERROR")
        raise FileNotFoundError(f"Tokenizer not found at {path}")

    tokenizer = Tokenizer.from_file(path)
    logger.log(f"Tokenizer loaded from {path}", level="SUCCESS")

    return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    """
    Encode text to token IDs.

    Args:
        tokenizer: The tokenizer to use
        text: Text to encode

    Returns:
        List[int]: Token IDs
    """
    encoding = tokenizer.encode(text)
    return encoding.ids


def decode(tokenizer: Tokenizer, token_ids: List[int]) -> List[str]:
    """
    Decode token IDs to list of token strings.

    Args:
        tokenizer: The tokenizer to use
        token_ids: List of token IDs to decode

    Returns:
        List[str]: List of token strings (e.g., ["hello", "world", "!"])
    """
    # Decode to text
    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Split text into tokens while preserving token boundaries
    # For ByteLevel tokens, we need to reconstruct based on Ġ markers
    tokens = []
    current_token = ""

    for char in text:
        if char == "Ġ":  # ByteLevel marker for space
            if current_token:
                tokens.append(current_token)
            current_token = ""
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    return tokens


def main() -> None:
    """Train and save tokenizer on processed shards."""
    logger = Logger(path="tokenizing.main")
    logger.log("Starting tokenization training...")

    try:
        # Build tokenizer
        tokenizer = build_tokenizer()

        # Train tokenizer
        train_tokenizer(tokenizer)

        # Save tokenizer
        save_path = Paths.TOKENIZER_FILE
        save_tokenizer(tokenizer, save_path)

        # Test encoding/decoding
        test_text = "Hello, world! This is a test."
        logger.log(f"Testing with: '{test_text}'")

        # Encode
        token_ids = encode(tokenizer, test_text)
        logger.log(f"Encoded token IDs: {token_ids[:10]}...")

        # Decode to list of strings
        token_strings = decode(tokenizer, token_ids)
        logger.log(
            f"Decoded tokens: {token_strings[:5]}...",
            level="SUCCESS",
        )

        # Show token count
        vocab_size = tokenizer.get_vocab_size()
        logger.log(f"Tokenizer vocab size: {vocab_size}", level="SUCCESS")

    except Exception as e:
        logger.log(f"Tokenization failed: {e}", level="ERROR")
        raise


if __name__ == "__main__":
    main()
