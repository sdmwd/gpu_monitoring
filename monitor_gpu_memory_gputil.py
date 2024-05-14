# pip install gputil
import GPUtil
import time
import os


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


try:
    while True:
        clear_console()
        gpus = GPUtil.getGPUs()

        for gpu in gpus:
            print(
                f"GPU {gpu.id}: {gpu.memoryUsed:.2f} MiB / {gpu.memoryTotal:.2f} MiB used")

        time.sleep(1)

except KeyboardInterrupt:
    print("Real-time monitoring stopped.")
