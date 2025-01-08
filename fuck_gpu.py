import time
import threading
import pynvml
import torch

pynvml.nvmlInit()
gpu_count = pynvml.nvmlDeviceGetCount()

# GPU usage threshold(%); tasks will run when usage is below this value
usage_threshold = 60.0
CHECK_INTERVAL = 10

running_flags = [False] * gpu_count

def get_gpu_usage(gpu_index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return utilization.gpu

def dummy_task(device, gpu_index):
    size = 1024  # Adjust matrix size as needed
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)

    while running_flags[gpu_index]:
        c = torch.mm(a, b)
        _ = c.sum()

def monitor_gpus():
    global running_flags

    try:
        while True:
            for i in range(gpu_count):
                # Pause the task first
                running_flags[i] = False
                time.sleep(1)
                usage = get_gpu_usage(i)
                device = torch.device(f"cuda:{i}")
                if usage < usage_threshold and not running_flags[i]:
                    running_flags[i] = True
                    threading.Thread(target=dummy_task, args=(device, i), daemon=True).start()
            
            time.sleep(CHECK_INTERVAL)

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    monitor_gpus()
