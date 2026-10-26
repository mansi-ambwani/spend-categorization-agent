@echo off
REM Starts the Streamlit web app at http://localhost:8501
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
