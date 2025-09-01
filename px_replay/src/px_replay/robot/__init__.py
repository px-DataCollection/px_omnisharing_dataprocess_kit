from colorama import init, Fore, Back, Style
import time
import threading
# from .tora import *
# from .robot_clerk import *
from .DexH13 import *
# from .controllers import *

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
    # 这里是实际的初始化代码
    # time.sleep(1)  # 模拟一些初始化工作
    # 可以添加更多实际的初始化步骤
    pass


def print_fancy_header(text):
    border = "+" + "=" * (len(text) + 2) + "+"
    print(Fore.CYAN + Style.BRIGHT + border)
    print(Fore.CYAN + Style.BRIGHT + f"| {text} |")
    print(Fore.CYAN + Style.BRIGHT + border)


def print_success_message():
    print(Fore.GREEN + Style.BRIGHT + "\n✔ Robot Module Initialized Successfully!")


print("\n")
print_fancy_header("ROBOT MODULE")

# # 创建一个事件来控制加载动画
# stop_event = threading.Event()

# # 在一个单独的线程中启动加载动画
# animation_thread = threading.Thread(target=print_loading_animation, args=(stop_event,))
# animation_thread.start()

# # 执行实际的初始化
# initialize_module()

# # 停止加载动画
# stop_event.set()
# animation_thread.join()

print_success_message()
print("\n")
