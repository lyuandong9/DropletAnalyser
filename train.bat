@echo off
set PYTHON_EXE=C:\Users\LYD\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo ========================================
echo   ImageIdentification - Model Training
echo ========================================
echo.
echo Available GPUs for training:
echo   1. NVIDIA RTX 2080 Ti (11 GB)
echo   2. NVIDIA RTX 5070 Ti (16 GB)  ^<-- Recommended
echo.
echo Make sure NVIDIA drivers and CUDA are installed.
echo.

cd /d "%~dp0"
"%PYTHON_EXE%" -c "from app.model.train import train_model, get_best_device; import sys; print('Data YAML path needed.'); print('Usage: python -c \"from app.model.train import train_model; train_model(data_yaml=\\\"dataset.yaml\\\", model_size=\\\"yolov8x-seg\\\", epochs=200, imgsz=1280, batch_size=8)\"')"
pause
