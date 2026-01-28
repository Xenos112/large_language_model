"""
To download data from the internet.
    - data used in this LLM is WikiPedia 20231101.en (https://huggingface.co/datasets/wikimedia/wikipedia)
    - data will be downloaded to the data folder in the ./data/raw
    - the data will be only downloaded and not processed (refer to ./src/data/process_data.py)

What does this code do?
    - Download the dataset from the internet.
    - Save the dataset into chunks.
    - Log the progress of the download.
"""

import os

from datasets import load_dataset
from tqdm import tqdm

from src.config.config import Paths
from src.utils.logger import Logger


# Function to download the dataset
def download_data():
    logger = Logger(file="download_data.py")

    logger.log(f"Downloading data... in {Paths.RAW_DATA_DIR}")

    os.makedirs(Paths.RAW_DATA_DIR, exist_ok=True)
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", streaming=False, split="train"
    )

    logger.log(f"Dataset loaded with size: {len(dataset)} articles")

    chunk_size = 10000
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc="Saving chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))

        output_file = os.path.join(
            Paths.RAW_DATA_DIR, f"wiki_chunk_{chunk_idx:04d}.txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for idx in range(start_idx, end_idx):
                article = dataset[idx]
                text = article.get("text", "")
                if text.strip():
                    f.write(text + "\n\n")

        logger.log(f"Saved chunk {chunk_idx + 1}/{num_chunks} to {output_file}")

    logger.log(f"Download complete! Data saved to {Paths.RAW_DATA_DIR}")
