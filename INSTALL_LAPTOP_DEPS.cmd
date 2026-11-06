@echo off
setlocal
echo This installs preparation tools into your LAPTOP Python, not the FPGA runtime.
python -m pip install -r "%~dp0requirements-laptop.txt"
if errorlevel 1 (
  echo Installation failed. Keep the error message above.
) else (
  echo Laptop dependencies ready. Now double-click PREPARE.cmd.
)
pause
