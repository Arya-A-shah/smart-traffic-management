@echo off
REM Windows: start the game backend (dashboard at http://127.0.0.1:8765)
cd /d "%~dp0"
python -c "import matplotlib" 2>nul || python -m pip install matplotlib numpy
python backend\server.py
pause
