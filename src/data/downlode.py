
import os
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from src.config.config import Paths
from utils.Logger import Logger


def download_data(chunk_size: int = 10000, streaming: bool = False) -> None:
    logger = Logger(path="download_data.download_data")
    logger.log(f"Downloading data to {Paths.RAW_DATA_DIR}")

    os.makedirs(Paths.RAW_DATA_DIR, exist_ok=True)
    
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en", 
            streaming=streaming, 
            split="train",
            trust_remote_code=True
        )
        
        if streaming:
            logger.log("Using streaming mode (memory efficient)")
            # For streaming, we process until we hit a limit or stream indefinitely
            _process_streaming(dataset, logger, chunk_size)
        else:
            logger.log(f"Dataset loaded with size: {len(dataset)} articles")
            _process_batched(dataset, logger, chunk_size)
            
        logger.log(
            f"Download complete! Data saved to {Paths.RAW_DATA_DIR}", 
            level="SUCCESS"
        )
        
    except Exception as e:
        logger.log(f"Download failed: {e}", level="ERROR")
        raise


def _process_batched(dataset, logger: Logger, chunk_size: int) -> None:
    """Process dataset in batches (non-streaming mode)."""
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="Saving chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
        
        output_file = os.path.join(
            Paths.RAW_DATA_DIR, 
            f"wiki_chunk_{chunk_idx:04d}.txt"
        )
        
        with open(output_file, "w", encoding="utf-8") as f:
            for idx in range(start_idx, end_idx):
                article = dataset[idx]
                text = article.get("text", "")
                if text.strip():
                    f.write(text + "\n\n")
                    
        logger.log(f"Saved chunk {chunk_idx + 1}/{num_chunks} to {output_file}")


def _process_streaming(dataset, logger: Logger, chunk_size: int) -> None:
    """Process dataset in streaming mode (low memory usage)."""
    chunk_idx = 0
    current_chunk = []
    
    for article in tqdm(dataset, desc="Streaming articles"):
        text = article.get("text", "")
        if text.strip():
            current_chunk.append(text)
            
        if len(current_chunk) >= chunk_size:
            _save_chunk(current_chunk, chunk_idx, logger)
            current_chunk = []
            chunk_idx += 1
            
    # Save remaining articles
    if current_chunk:
        _save_chunk(current_chunk, chunk_idx, logger)


def _save_chunk(articles: list, chunk_idx: int, logger: Logger) -> None:
    """Save a chunk of articles to disk."""
    output_file = os.path.join(
        Paths.RAW_DATA_DIR, 
        f"wiki_chunk_{chunk_idx:04d}.txt"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(articles))
        
    logger.log(f"Saved chunk {chunk_idx} to {output_file}")


if __name__ == "__main__":
    download_data()
