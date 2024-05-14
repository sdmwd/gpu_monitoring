@echo off
REM Activate the virtual environment
call conda activate n:/TESTS/.conda
:loop
cls
REM Run the Python script
python monitor_gpu_memory_gputil.py
if %errorlevel% neq 0 (
    echo An error occurred. Exiting...
    exit /b %errorlevel%
)
goto loop
REM Deactivate the virtual environment
call conda deactivate