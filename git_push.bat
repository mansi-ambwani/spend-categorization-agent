@echo off
REM Commits this project and pushes it to https://github.com/mansi-ambwani/spend-categorization-agent
cd /d "%~dp0"
echo ==== %date% %time% ==== > git_log.txt
git --version >> git_log.txt 2>&1
if errorlevel 1 (echo GIT_NOT_INSTALLED >> git_log.txt & goto end)
if not exist .git git init -b main >> git_log.txt 2>&1
set GN=
for /f "delims=" %%a in ('git config user.name') do set GN=%%a
if not defined GN git config user.name "Mansi Ambwani"
set GE=
for /f "delims=" %%a in ('git config user.email') do set GE=%%a
if not defined GE git config user.email "mansi-ambwani@users.noreply.github.com"
git add -A >> git_log.txt 2>&1
git commit -q -m "Automated spend categorization with Llama-3 8B QLoRA and a natural-language spend agent" -m "Synthetic procurement dataset, TF-IDF baseline, QLoRA fine-tuning (Colab/Kaggle), evaluation, batch categorization, SQL spend agent and Streamlit app." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01MXJyb74goH2bkjsK3fLQ1w" >> git_log.txt 2>&1
git branch -M main >> git_log.txt 2>&1
git remote remove origin >nul 2>&1
git remote add origin https://github.com/mansi-ambwani/spend-categorization-agent.git >> git_log.txt 2>&1
git push -u origin main >> git_log.txt 2>&1
echo PUSH_EXIT=%errorlevel% >> git_log.txt
git log --oneline -3 >> git_log.txt 2>&1
:end
type git_log.txt
timeout /t 30
