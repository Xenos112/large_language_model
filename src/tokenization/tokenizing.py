"""
BPE Tokenizer that trains on processed shards from Paths.PROCESSED_DATA_DIR
    - Saves to Paths.TOKENIZER_FILE in JSON format
"""

import os
import glob
from typing import List, Optional, Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

from src.config.config import Paths, ModelConfig
from src.utils.logger import Logger


class BPETokenizer:
    def __init__(self):
        self.logger = Logger(__name__)
        self.tokenizer = None
        self._ensure_directories()
        
    def _ensure_directories(self):
        os.makedirs(Paths.DATA_DIR, exist_ok=True)
        os.makedirs(Paths.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(Paths.PROCESSED_DATA_DIR, exist_ok=True)
        
        tokenizer_dir = os.path.dirname(Paths.TOKENIZER_FILE)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
            
    def get_processed_shards(self) -> List[str]:
        if not os.path.exists(Paths.PROCESSED_DATA_DIR):
            self.logger.log(f"Processed data dir not found: {Paths.PROCESSED_DATA_DIR}", level="WARNING")
            return []
        
        patterns = [
            os.path.join(Paths.PROCESSED_DATA_DIR, "*.txt"),
            os.path.join(Paths.PROCESSED_DATA_DIR, "*.shard"),
            os.path.join(Paths.PROCESSED_DATA_DIR, "shard_*"),
            os.path.join(Paths.PROCESSED_DATA_DIR, "data_*"),
        ]
        
        shards = []
        for pattern in patterns:
            shards.extend(glob.glob(pattern))
        
        shards = sorted(list(set(shards)))  # Remove duplicates, sort
        
        if shards:
            self.logger.log(f"Found {len(shards)} shard(s) in {Paths.PROCESSED_DATA_DIR}", "INFO")
            for shard in shards[:5]:
                self.logger.log(f"  - {os.path.basename(shard)}", "DEBUG")
            if len(shards) > 5:
                self.logger.log(f"  ... and {len(shards) - 5} more", "DEBUG")
        else:
            self.logger.log(f"No shards found in {Paths.PROCESSED_DATA_DIR}", "WARNING")
            
        return shards
    
    def create_tokenizer(self, vocab_size: Optional[int] = None) -> Tokenizer:
        vocab_size = vocab_size or ModelConfig.vocab_size
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = Sequence([
            NFD(),
            StripAccents(),
            Lowercase()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.enable_truncation(max_length=ModelConfig.max_sequence_length)
        
        return tokenizer
    
    def train(self, files: Union[str, List[str], None] = None, vocab_size: Optional[int] = None) -> None:
        if files is None:
            files = self.get_processed_shards()
            if not files:
                raise FileNotFoundError(f"No shards found in {Paths.PROCESSED_DATA_DIR}")
        elif isinstance(files, str):
            files = [files]
            
        vocab_size = vocab_size or ModelConfig.vocab_size
        
        self.logger.log(f"Training on {len(files)} shard(s)...", "INFO")
        self.logger.log(f"Target vocab: {vocab_size}, Max length: {ModelConfig.max_sequence_length}", "INFO")
        
        self.tokenizer = self.create_tokenizer(vocab_size)
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
            min_frequency=2,
            show_progress=True
        )
        
        self.tokenizer.train(files, trainer)
        
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )
        
        self.logger.log(f"Training complete. Vocab size: {self.get_vocab_size()}", "SUCCESS")
        
    def save(self, path: Optional[str] = None) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save. Train or load first.")
            
        save_path = path or Paths.TOKENIZER_FILE
        
        # Ensure .json extension
        if not save_path.endswith('.json'):
            save_path += '.json'
            
        self.tokenizer.save(save_path)
        
        # Verify
        if os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            self.logger.log(f"Saved JSON to {save_path} ({size_mb:.2f} MB)", "SUCCESS")
        
        return save_path
        
    def load(self, path: Optional[str] = None) -> None:
        load_path = path or Paths.TOKENIZER_FILE
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer not found: {load_path}")
            
        self.tokenizer = Tokenizer.from_file(load_path)
        self.logger.log(f"Loaded tokenizer from {load_path}", "SUCCESS")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    
    def __len__(self) -> int:
        return self.get_vocab_size()


# Convenience functions
def train_tokenizer_on_shards(vocab_size: Optional[int] = None):
    tokenizer = BPETokenizer()
    tokenizer.train(vocab_size=vocab_size)
    tokenizer.save()
    return tokenizer


def load_tokenizer(path: Optional[str] = None):
    tokenizer = BPETokenizer()
    tokenizer.load(path)
    return tokenizer


if __name__ == "__main__":
    tokenizer = train_tokenizer_on_shards()
    
    ids = tokenizer.encode("hello world")
    print(f"Tokenized: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")
