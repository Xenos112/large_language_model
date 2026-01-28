"""
Logging utilities with colorized console output.
"""

from datetime import datetime
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
        "SUCCESS": (Fore.CYAN, "SUCCESS"),  # Fixed: was showing as DEBUG
        "DEBUG": (Fore.MAGENTA, "DEBUG")
    }
    
    def __init__(self, path: str, log_to_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            path: Module path for log prefix
            log_to_file: Optional file path to also write logs to
        """
        self.path = path
        self.log_to_file = log_to_file
        
        if log_to_file:
            Path(log_to_file).parent.mkdir(parents=True, exist_ok=True)

    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with colorized output.
        
        Args:
            message: The message to log
            level: INFO, WARNING, ERROR, SUCCESS, or DEBUG
        """
        color, label = self.LEVEL_COLORS.get(level, (Fore.WHITE, level))
        
        # Console output with colors
        print(
            f"{self.get_timestamp()} {color}{Style.BRIGHT}[{label}]{Style.RESET_ALL}"
            f"[{self.path}] {message}"
        )
        
        # Optional file logging (no colors)
        if self.log_to_file:
            try:
                with open(self.log_to_file, "a") as f:
                    f.write(f"{self.get_timestamp()} [{label}][{self.path}] {message}\n")
            except Exception:
                pass  # Fail silently for file logging

    def info(self, message: str) -> None:
        """Convenience method for INFO level."""
        self.log(message, "INFO")
        
    def warning(self, message: str) -> None:
        """Convenience method for WARNING level."""
        self.log(message, "WARNING")
        
    def error(self, message: str) -> None:
        """Convenience method for ERROR level."""
        self.log(message, "ERROR")
        
    def success(self, message: str) -> None:
        """Convenience method for SUCCESS level."""
        self.log(message, "SUCCESS")


# Add missing import for Path
from pathlib import Path
