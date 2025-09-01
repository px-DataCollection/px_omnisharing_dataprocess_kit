from px_replay.robot import *
from px_replay.dataprocess import *

# from utils import *
from colorama import init, Fore, Back, Style
import time
import threading

init(autoreset=True)  # 初始化 colorama

def print_loading_animation(stop_event):
    start_time = time.time()

    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        bar_length = 20
        filled_length = int(bar_length * (elapsed_time % 1))
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        print(
            Fore.YELLOW + f"\rLoading: [{bar}] {elapsed_time:.1f}s", end="", flush=True
        )
        time.sleep(0.1)

    print(
        Fore.GREEN
        + f"\nLoading completed in {elapsed_time:.1f} seconds."
        + Style.RESET_ALL
    )

def initialize_module():
    pass


def print_fancy_header(text):
    border = "+" + "=" * (len(text) + 2) + "+"
    print(Fore.CYAN + Style.BRIGHT + border)
    print(Fore.CYAN + Style.BRIGHT + f"| {text} |")
    print(Fore.CYAN + Style.BRIGHT + border)


def print_success_message():
    print(
        Fore.GREEN + Style.BRIGHT + "\n✔ PX Replay Initialized Successfully!"
    )

print("\n")
print_fancy_header("px_replay")

from pyfiglet import Figlet
from colorama import init, Fore, Style

init(autoreset=True)

f = Figlet(font="roman")
text = f.renderText("PaXini EID")

print(Fore.YELLOW + text)

waterMarkstr = """                         
Author: Filippo Pacini
Version: 1.0.0a
Copyright (c) 2025 Paxini
All rights reserved.
"""
print(waterMarkstr)

print_success_message()
print("\n")
