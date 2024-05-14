# GPU Monitoring

This repository contains scripts and utilities for monitoring GPU memory usage in real-time using Python.

## Contents

- `gpu_monitoring.yml` - Conda environment configuration file.
- `check_cuda_cudnn` - Python script for checking CUDA - CUDNN installation
- `monitor_gpu_memory_pynvml.py` - Python script for monitoring GPU memory usage in real-time using `pynvml`.
- `monitor_gpu_memory_gputil.py` - Python script for monitoring GPU memory usage in real-time using `gputil`.
- `run_monitor_XXXXXX.bat` - Batch files to run the Python scripts.
- `run_monitor_XXXXXX_conda.bat` - Batch files to activate the conda environment and run the Python scripts.


## Setup

### Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with proper drivers installed

### Installation

1. **Create and Activate the Conda Environment:**

   ```bash
   conda env create -f gpu_monitoring.yml
   conda activate gpu_monitoring
