"""
Contains Base config for the LLM including Paths...
TODO: ADD MORE COMMENTS
"""


class Paths:
    DATA_DIR = "./data"
    RAW_DATA_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"


class SaveData:
    shard_size = 1024 * 1024 * 1024  # 1GB
