@echo off
REM Full pipeline for Windows. Run from the project folder.
set PYTHONPATH=%~dp0
python -m sim.build_network || exit /b
python -m forecast.collect_data --episodes 12 || exit /b
python -m forecast.train_forecaster --epochs 40 || exit /b
python -m rl.train --episodes 100 || exit /b
python -m rl.train --episodes 100 --forecast || exit /b
python -m rl.train --episodes 100 --algo mappo --forecast || exit /b
python -m evaluate.benchmark || exit /b
echo Done. Start the dashboard with:  streamlit run dashboard/app.py
