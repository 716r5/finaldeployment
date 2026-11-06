@echo off
setlocal
cd /d "%~dp0"
echo Select one or multiple additional real HSI MAT images.
python prepare_queue.py --additional
if errorlevel 1 (
  echo Preparation failed. Keep the error above.
) else (
  echo Copy the additional_inputs queue folder shown above to the USB.
  echo Import it on the board using the exact command in README.md.
)
pause
