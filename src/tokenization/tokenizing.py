"""
Core of the LLM: The Tokenizer
    - this file will handle the tokenization using the rustbpe library
"""

from pathlib import Path
import json

import rustbpe

from src.config.config import ModelConfig, Paths
from src.utils.logger import Logger


def shard_iterator():
    logger = Logger(path="tokenizing.shard_iterator")
    shards = sorted(Path(Paths.PROCESSED_DATA_DIR).glob("*.txt"))

    for shard in shards:
        logger.log(f"Tokenizing with {shard}")
        with open(shard, "r", encoding="utf-8") as file:
            content = file.read().split('\n\n')
            for line in content
                yield line


def main():
    logger = Logger(path="tokenizing.main")
    logger.log("Start Tokenizing...")

    # FIX: its throwing error for no reason (https://github.com/karpathy/rustbpe)
    tokenizer = rustbpe.Tokenizer()
    try:
        tokenizer.train_from_iterator(
            shard_iterator(),
            vocab_size=ModelConfig.vocab_size
        )
        tokenizer_state = {
            "vocab_size": tokenizer.vocab_size,
            "pattern": tokenizer.get_pattern(),
            "version": "rustbpe_v1"
        }
        logger.log("Tokenization complete!", level="SUCCESS")
        logger.log("Save in a file")
        with open(Paths.TOKENIZER_FILE, "w", encoding="utf-8") as file:
            file.write(json.dumps(tokenizer_state))

        logger.log(f"Saved: {Paths.TOKENIZER_FILE}", level="SUCCESS")
    except Exception as e:
        logger.log(f"Error occurred during tokenization: {e}", level="ERROR")
