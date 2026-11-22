@echo off
cd /d "%~dp0"
python -m pip install -r requirements-desktop.txt --quiet
start "" /min python desktop_app.py
exit
