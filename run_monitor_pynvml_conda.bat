@echo off
REM Activate the virtual environment
call conda activate n:/TESTS/.conda
REM Run the Python script
python n:/TESTS/monitor_gpu_memory_pynvml.py
REM Deactivate the virtual environment
call conda deactivate
pause
