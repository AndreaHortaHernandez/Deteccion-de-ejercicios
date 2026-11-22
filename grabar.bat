@echo off
rem Doble clic para grabar los 9 videos de una persona con la webcam.
rem Los videos quedan en videos\apellido_nombre\ y las reps en datos\conteos_reales.csv
cd /d "%~dp0"
py -3 grabar_webcam.py
echo.
pause
