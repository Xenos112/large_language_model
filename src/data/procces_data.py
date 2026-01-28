

import hashlib
import os
import re
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Set

from tqdm import tqdm

from src.config.config import Paths, SaveData
from utils.Logger import Logger


def clean_text(text: str) -> str:
    """
    Clean and normalize Wikipedia text.
    
    Operations:
        - Remove HTML/XML tags
        - Remove Wiki markup ([[...]])
        - Normalize whitespace
        - Remove non-ASCII characters (optional, can be removed if multilingual needed)
    """
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove Wiki markup like [[Category:...]] or [[File:...]]
    text = re.sub(r"\[\[.*?\]\]", "", text)
    
    # Normalize whitespace (newlines, tabs -> single space)
    text = re.sub(r"\s+", " ", text)
    
    # Remove non-printable characters but keep basic unicode for now
    # text = re.sub(r"[^\x00-\x7F]+", "", text)  # Commented out: removes non-ASCII
    
    return text.strip()


def hash_text(text: str) -> str:
    """Generate SHA256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StreamingShardWriter:  # Fixed: Writer not Writter
    """
    Manages writing text to sharded files with size limits.
    Handles deduplication and automatic shard rotation.
    """
    
    def __init__(self, shard_size: int = SaveData.shard_size) -> None:
        self.logger = Logger(path="process_data.StreamingShardWriter")
        self.shard_size = shard_size
        self.shard_index = 0
        self.current_file: Optional[TextIOWrapper] = None  # Fixed: removed field()
        self.hashes_seen: Set[str] = set()  # Fixed: hashes not hashs
        self.current_size = 0
        self.total_written_texts = 0
        self.duplicates_skipped = 0
        
        # Ensure directory exists
        os.makedirs(Paths.PROCESSED_DATA_DIR, exist_ok=True)
        self.open_new_shard()
        
        self.logger.log("Initialized StreamingShardWriter")

    def open_new_shard(self) -> None:
        """Close current shard (if any) and open a new one."""
        if self.current_file:
            self.current_file.close()
            self.logger.log(f"Closed shard {self.shard_index - 1}")
            
        shard_path = os.path.join(
            Paths.PROCESSED_DATA_DIR, 
            f"shard_{self.shard_index:04d}.txt"
        )
        
        self.current_file = open(shard_path, "w", encoding="utf-8", buffering=8192)
        self.current_size = 0  # Reset size counter
        self.shard_index += 1
        
        self.logger.log(f"Opened new shard: {shard_path}")

    def should_rotate_shard(self, text_size: int) -> bool:
        """Determine if we need to rotate to a new shard."""
        return (self.current_size + text_size) > self.shard_size

    def add_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
            
        # Check for duplicates
        text_hash = hash_text(text)
        if text_hash in self.hashes_seen:
            self.duplicates_skipped += 1
            return False
            
        self.hashes_seen.add(text_hash)
        
        # Calculate size with delimiter
        text_bytes = text.encode("utf-8")
        delimiter_size = 2 if self.current_size > 0 else 0  # \n\n for non-first items
        text_size = len(text_bytes) + delimiter_size
        
        # Rotate shard if needed
        if self.should_rotate_shard(text_size):
            self.open_new_shard()
            
        # Write to file
        if self.current_size > 0:
            self.current_file.write("\n\n")
            
        self.current_file.write(text)
        self.current_size += text_size
        self.total_written_texts += 1
        return True

    def close(self) -> None:
        """Close current shard and log stats."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None  # Fixed: Added None
            
        self.logger.log(f"Closed shard {self.shard_index - 1}")
        self.logger.log(f"Total texts written: {self.total_written_texts}")
        self.logger.log(f"Duplicates skipped: {self.duplicates_skipped}")
        self.logger.log(f"Total shards created: {self.shard_index}")
        
        # Optional: Save hash set to disk for resumability
        hash_cache_path = os.path.join(Paths.PROCESSED_DATA_DIR, "hashes.cache")
        try:
            with open(hash_cache_path, "w") as f:
                for h in self.hashes_seen:
                    f.write(h + "\n")
            self.logger.log(f"Saved hash cache to {hash_cache_path}")
        except Exception as e:
            self.logger.log(f"Failed to save hash cache: {e}", level="WARNING")

    def __enter__(self):
        """Context manager support."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup on exit."""
        self.close()
        return False


def process_articles(writer: StreamingShardWriter, file_path: Path) -> int:
    written = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Split by double newlines (article separator)
    articles = content.split('\n\n')  # Fixed spelling: articles not articales
    
    for article in articles:
        cleaned = clean_text(article)
        
        # Only keep substantial articles (100+ chars)
        if len(cleaned) >= 100:
            if writer.add_text(cleaned):
                written += 1
                
    return written


def main() -> None:
    """Main preprocessing pipeline."""
    logger = Logger(path="process_data.main")
    logger.log("Starting data preprocessing...")

    raw_files = sorted(Path(Paths.RAW_DATA_DIR).glob("*.txt"))
    
    if not raw_files:
        logger.log("No raw files found in " + Paths.RAW_DATA_DIR, level="ERROR")
        return
        
    logger.log(f"Found {len(raw_files)} files to process")
    
    # Use context manager for safe cleanup
    with StreamingShardWriter() as writer:
        total_articles = 0
        
        for raw_file in tqdm(raw_files, desc="Processing Files"):
            try:
                count = process_articles(writer, raw_file)
                total_articles += count
                logger.log(f"Processed {raw_file.name}: {count} articles written")
            except Exception as e:
                logger.log(f"Error processing {raw_file}: {e}", level="ERROR")
                # Continue with next file instead of crashing
                continue
                
    logger.log(f"Preprocessing complete! Total articles: {total_articles}", level="SUCCESS")


if __name__ == "__main__":
    main()
