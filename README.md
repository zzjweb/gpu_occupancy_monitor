# gpu_occupancy_monitor
Monitor GPU usage, occupy it when usage is below the threshold.监视gpu使用率，低于阈值就占满，防止kill

### Description

A script for occupying GPUs: Monitors GPU utilization and performs computations when GPU utilization is below a specified threshold to prevent being killed due to insufficient utilization.

### Features
- Continuously monitors GPU usage using NVIDIA's NVML library.
- Automatically starts dummy matrix multiplication tasks on GPUs with low utilization.
- Stops dummy tasks when utilization rises above the specified threshold.
- Supports multi-GPU systems.
- Uses PyTorch for GPU computation tasks.

### Requirements
- Python 3.x
- NVIDIA GPU and drivers
- PyTorch
- `nvidia-ml-py3` library (install using `pip install nvidia-ml-py3`)

### How It Works
1. The script initializes NVML to monitor GPU utilization.
2. It checks the utilization of each GPU at regular intervals.
3. If a GPU's utilization is below the threshold (default: 60%), it starts a dummy task that performs matrix multiplications on that GPU.
4. If the utilization exceeds the threshold, the dummy task is stopped.

### Usage
1. Install the required libraries:
   ```bash
   pip install torch nvidia-ml-py3
   ```
2. Run the script:
   nohup python3 fuck_gpu.py

---

### 描述

占卡脚本，监控 GPU 利用率，并在 GPU 利用率低于指定阈值时计算，防止利用率不足被kill。


### 环境需求
- Python 3.x
- NVIDIA GPU 和驱动
- PyTorch
- `nvidia-ml-py3` 库（安装命令：`pip install nvidia-ml-py3`）


### 使用方法
1. 安装所需库：
   ```bash
   pip install torch nvidia-ml-py3
   ```
2. 运行脚本：
   nohup python3 fuck_gpu.py
