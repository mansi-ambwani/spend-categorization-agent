@echo off
REM One-time setup on Windows: creates a virtual environment and installs the app requirements.
cd /d "%~dp0"
python --version || (echo Python is not installed. Install Python 3.11+ from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
if not exist models\baseline_tfidf_logreg.joblib python -m src.baseline
if not exist data\outputs\categorized_spend.csv python -m src.categorize --model baseline
python -m pytest -q -k "not tiny_llm"
echo.
echo Setup finished. Double-click run_app.bat to start the app.
pause
