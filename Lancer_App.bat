@echo off
title Energy Management System
cd /d "%~dp0"
echo Lancement de l application... (ne ferme pas cette fenetre)
python -m streamlit run app.py
pause
