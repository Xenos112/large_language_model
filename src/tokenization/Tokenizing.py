

import json
from pathlib import Path
from typing import Iterator

import rustbpe

from src.config.config import ModelConfig, Paths
from utils.Logger import Logger


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
                articles = content.split('\n\n')
                
                for line in articles:  # Fixed: Added missing colon
                    if line.strip():  # Skip empty lines
                        yield line
        except Exception as e:
            logger.log(f"Error reading {shard}: {e}", level="WARNING")
            continue


def save_tokenizer(tokenizer, path: str) -> None:
    """Save tokenizer state and merge rules."""
    # Get vocabulary
    vocab = tokenizer.get_vocab()
    
    tokenizer_state = {
        "vocab_size": tokenizer.vocab_size,
        "vocab": vocab,
        "pattern": tokenizer.get_pattern(),
        "version": "rustbpe_v1",
        "special_tokens": {
            "<|endoftext|>": vocab.get("<|endoftext|>", 0),
            "<|pad|>": vocab.get("<|pad|>", 1),
        }
    }
    
    with open(path, "w", encoding="utf-8") as file:
        json.dump(tokenizer_state, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Train tokenizer on processed shards."""
    logger = Logger(path="tokenizing.main")
    logger.log("Starting tokenization training...")
    
    # Check if rustbpe is properly installed
    try:
        tokenizer = rustbpe.Tokenizer()
    except Exception as e:
        logger.log(f"Failed to initialize rustbpe tokenizer: {e}", level="ERROR")
        logger.log("Ensure rustbpe is installed: pip install rustbpe", level="ERROR")
        return
    
    try:
        logger.log(f"Training with vocab_size={ModelConfig.vocab_size}")
        
        # Train from iterator
        tokenizer.train_from_iterator(
            shard_iterator(),
            vocab_size=ModelConfig.vocab_size,
            min_frequency=2  # Optional: ignore very rare tokens
        )
        
        logger.log("Training complete!", level="SUCCESS")
        
        # Save tokenizer
        save_path = Paths.TOKENIZER_FILE
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        save_tokenizer(tokenizer, save_path)
        logger.log(f"Tokenizer saved to {save_path}", level="SUCCESS")
        
        # Test encoding/decoding
        test_text = "Hello, world! This is a test."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        logger.log(f"Test encode/decode: '{test_text}' -> {encoded[:5]}... -> '{decoded}'")
        
    except Exception as e:
        logger.log(f"Tokenization failed: {e}", level="ERROR")
        raise


if __name__ == "__main__":
    main()
