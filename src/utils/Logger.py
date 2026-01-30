"""
Logging utilities with colorized console output.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from colorama import Fore, Style


class Logger:
    """
    Simple colored logger for console output.
    """

    LEVEL_COLORS = {
        "INFO": (Fore.GREEN, "INFO"),
        "WARNING": (Fore.YELLOW, "WARNING"),
        "ERROR": (Fore.RED, "ERROR"),
        "SUCCESS": (Fore.CYAN, "SUCCESS"),
        "DEBUG": (Fore.MAGENTA, "DEBUG"),
    }

    def __init__(self, path: str):
        self.path = path

    def get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message: str, level: str = "INFO") -> None:
        color, label = self.LEVEL_COLORS.get(level, (Fore.WHITE, level))

        print(
            f"{self.get_timestamp()} {color}{Style.BRIGHT}[{label}]{Style.RESET_ALL} [{self.path}] {message}"
        )
