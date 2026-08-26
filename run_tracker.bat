@echo off
title Sigara Takip Sistemi
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" face_and_hands_tracker.py
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" face_and_hands_tracker.py
) else (
    python face_and_hands_tracker.py
)

pause
