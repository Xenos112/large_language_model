from datetime import datetime

from colorama import Fore, Style


class Logger:
    def __init__(self, path: str):
        self.log_level = "INFO"
        self.path = path

    def get_date(self):
        return datetime.now()

    def log(self, message, level="INFO"):
        if level == "INFO":
            print(
                f"{self.get_date()} {Fore.GREEN}{Style.BRIGHT}[INFO]{Style.RESET_ALL}[{self.path}] {message}"
            )
        elif level == "WARNING":
            print(
                f"{self.get_date()} {Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}[{self.path}] {message}"
            )
        elif level == "ERROR":
            print(
                f"{self.get_date()} {Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL}[{self.path}] {message}"
            )
        elif level == "SUCCESS":
            print(
                f"{self.get_date()} {Fore.BLUE}{Style.BRIGHT}[DEBUG]{Style.RESET_ALL}[{self.path}] {message}"
            )
