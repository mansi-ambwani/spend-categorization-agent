@echo off
REM Sends the Llama-3 QLoRA training job to Kaggle (free T4 GPU). It keeps running even if this laptop sleeps.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ==== push %date% %time% ==== > kaggle_log.txt
pip install -q kaggle >> kaggle_log.txt 2>&1
python kaggle_job\make_metadata.py >> kaggle_log.txt 2>&1
if errorlevel 1 goto fail
kaggle kernels push -p kaggle_job --accelerator NvidiaTeslaT4 >> kaggle_log.txt 2>&1
if errorlevel 1 kaggle kernels push -p kaggle_job >> kaggle_log.txt 2>&1
type kaggle_log.txt
echo.
echo Job sent to Kaggle. In about an hour, double-click kaggle_results.bat to download the results.
timeout /t 20
exit /b 0
:fail
type kaggle_log.txt
timeout /t 60
