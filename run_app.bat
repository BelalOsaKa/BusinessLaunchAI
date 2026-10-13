@echo off
rem Starts the BusinessLaunch AI app from the project root so the theme in .streamlit/config.toml is used.
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "..\..\.venv\Scripts\python.exe" set "PY=..\..\.venv\Scripts\python.exe"
"%PY%" -m streamlit run app\Home.py %*
