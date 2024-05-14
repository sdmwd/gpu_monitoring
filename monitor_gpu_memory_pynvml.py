import pynvml
import time
import os


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

pynvml.nvmlInit()

try:
    while True:
        clear_console()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(
                f"GPU {i}: {mem_info.used / 1024**2:.2f} MiB / {mem_info.total / 1024**2:.2f} MiB used")

        time.sleep(1)

except KeyboardInterrupt:
    print("Real-time monitoring stopped.")

finally:
    pynvml.nvmlShutdown()
