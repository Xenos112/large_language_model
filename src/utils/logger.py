from datetime import datetime

from colorama import Fore, Style


class Logger:
    def __init__(self, file: str):
        self.log_level = "INFO"
        self.file = file

    def get_date(self):
        return datetime.now()

    def log(self, message, level="INFO"):
        if level == "INFO":
            print(
                f"{self.get_date()} {Fore.GREEN}{Style.BRIGHT}[INFO]{Style.RESET_ALL}[{self.file}] {message}"
            )
        elif level == "WARNING":
            print(
                f"{self.get_date()} {Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}[{self.file}] {message}"
            )
        elif level == "ERROR":
            print(
                f"{self.get_date()} {Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL}[{self.file}] {message}"
            )
        elif level == "DEBUG":
            print(
                f"{self.get_date()} {Fore.BLUE}{Style.BRIGHT}[DEBUG]{Style.RESET_ALL}[{self.file}] {message}"
            )
