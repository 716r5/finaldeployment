@echo off
setlocal
cd /d "%~dp0"
echo Oil-spill application preparation. Original USB files are not changed.
for %%F in (oil_spill_cnn_138.xmodel test_patch.npy test_patch.json) do (
  if not exist "%%F" (
    echo MISSING: %%F. Copy it from your previous deploy folder into this NEW folder.
    pause
    exit /b 1
  )
)
python -c "import numpy,scipy,PIL,tkinter" >nul 2>nul
if errorlevel 1 (
  echo Laptop dependencies missing. Double-click INSTALL_LAPTOP_DEPS.cmd, then retry PREPARE.cmd.
  pause
  exit /b 1
)
if exist scene_queue.json goto seal
python prepare_queue.py
if errorlevel 1 goto failed
:seal
if exist release.json goto verify
python deploy.py seal >> preparation-log.txt 2>&1
if errorlevel 1 goto failed
:verify
python deploy.py verify >> preparation-log.txt 2>&1
if errorlevel 1 goto failed
echo.
echo READY TO TAKE TO THE LAB: the real scene and model are packaged.
echo Classification will run on the FPGA. No prediction has been made on this laptop.
echo Copy this whole NEW folder onto the USB, keeping the old project unchanged.
pause
exit /b 0
:failed
echo Preparation failed. Keep the error above. Do not change the original project.
if exist preparation-log.txt type preparation-log.txt
pause
exit /b 1
