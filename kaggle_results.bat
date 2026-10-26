@echo off
REM Checks the Kaggle training job; when it has finished, downloads results.zip and unpacks it into this project.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set /p KUSER=<kaggle_job\kaggle_user.txt
echo ==== status %date% %time% ==== > kaggle_status.txt
kaggle kernels status %KUSER%/spend-categorization-llama3-qlora >> kaggle_status.txt 2>&1
findstr /i "complete" kaggle_status.txt >nul
if errorlevel 1 goto notdone
kaggle kernels output %KUSER%/spend-categorization-llama3-qlora -p kaggle_output >> kaggle_status.txt 2>&1
tar -xf kaggle_output\results.zip -C . >> kaggle_status.txt 2>&1
if errorlevel 1 goto notdone
echo RESULTS_INSTALLED >> kaggle_status.txt
:notdone
type kaggle_status.txt
timeout /t 15
