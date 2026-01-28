"""
What Does this code do?
    - Process the downloaded dataset.
    - Split the dataset into chunks.
    - Log the progress of the processing.
"""

from codecs import utf_8_decode
from dataclasses import field
import hashlib
import os
import re
from io import TextIOWrapper
from turtle import st
from typing import Optional

from tqdm import tqdm
from pathlib import Path

from src.config.config import Paths
from src.utils.logger import Logger


def clean_text(text: str) -> str:
    """
    What does this function do?
        - clean the dataset
        - remove links
        - remove wiki markups
        - remove white spaces
        - remove non-printed characters
    """
    logger = Logger(file="process_data.py")
    logger.log("Processing data...")

    # Remove URLs
    text = re.sub(r"<[^>]+>", "", text)

    # Remove Wiki markups
    text = re.sub(r"\[\[.*?\]\]", "", text)

    # Remove white spaces
    text = re.sub(r"\s+", " ", text)

    # Remove non-printed characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    return text.strip()


def hash_text(text: str) -> str:
    """
    What does this function do?
        - hash the text
    Why?
        - save hashed to avoid data redundency
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class StreamingShardWritter:
    """
    What does this class do?
        - stream text into fixed sized shards
    """

    def __init__(self) -> None:
        self.logger = Logger(file="process_data.py")
        self.shard_index = 0  # index for shards
        self.current_file: Optional[TextIOWrapper] = field(default=None)  # see if we have open shard
        self.hashs_seen = set()  # list of hashes to avoid data redundency
        self.current_size = 0  # needed to see if we can add more text to the shard
        self.total_written_texts = 0  # total number of texts written

        self.logger.log(f"Make Directory: {Paths.PROCESSED_DATA_DIR}")
        os.makedirs(Paths.PROCESSED_DATA_DIR, exist_ok=True)

    def open_new_shard(self) -> None:
        if self.current_file:
            self.current_file.close()
            self.logger.log(f"Closed shard {self.shard_index - 1}")
        shard_path = os.path.join(
            Paths.PROCESSED_DATA_DIR, f"shard_{self.shard_index}.txt"
        )
        self.current_file = open(shard_path, "w", encoding="utf-8")
        self.shard_index += 1

    def add_text(self, text: str) -> bool:
        text_hash = hash_text(text)
        if text_hash in self.hashs_seen:
            return False
        self.hashs_seen.add(text_hash)
        text_bytes = text.encode("utf-8")
        text_size = (
            len(text_bytes) + 2
        )  # +2 just cuz there is /n/n, they are treated as two bytes
        if self.current_size > 0:
            self.current_file.write("\n\n") # FIX: i dont know how to get rid of this error

        self.current_file.write(text)
        self.current_size += text_size
        self.total_written_texts += 1
        return True

    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file =
        self.logger.log(f"Closed shard {self.shard_index - 1}")
        self.logger.log(f"Total texts written: {self.total_written_texts}")
        self.logger.log(f"Total shards created: {self.shard_index}")

def main():
    """
        - The main function for preprocessing data
        - the end result is text files each with 1GB of size
    """
    logger = Logger(file="process_data.py")
    logger.log("Starting data preprocessing...")
    logger.log("Getting Raw Files")

    raw_files = sorted(Path(Paths.RAW_DATA_DIR).glob("*.txt"))
    if len(raw_files) == 0:
        logger.log("No files found, Exiting Process Now",level="ERROR")
        return

    logger.log(f"Found {len(raw_files)} files")
    stream_writter = StreamingShardWritter()
    text_buffers = []

    try:
        for raw_file in tqdm(raw_files,desc="Processing Files"):
            with open(raw_file, 'r', encoding='utf-8') as file:
                file_content = file.read()

            articales = file_content.split('\n\n')
            for article in articales:
                cleaned_article = clean_text(article)

                if len(cleaned_article) >= 100:
                    text_buffers.append(cleaned_article)
                    if len(text_buffers) >= 1000:
                        for text in text_buffers:
                            stream_writter.add_text(text)
                            text_buffers = []
            del articales, file_content # Cuz the memory will be filled

        if text_buffers:
            for text in text_buffers:
                stream_writter.add_text(text)
            text_buffers.clear()
    except Exception as e:
        logger.log(f"Error occurred while processing files: {e}", level="ERROR")
        raise e
    finally:
        stream_writter.close()

    logger.log("Data preprocessing completed",level="SUCCESS")

if __name__ == "__main__":
    main()
