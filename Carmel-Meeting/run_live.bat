@echo off
setlocal
cd /d "%~dp0"

echo.
echo Carmel Meeting LIVE Transcription (local mic)
echo ---------------------------------------------
echo Speaks into the mic; text appears after short pauses.
echo Existing batch tool (run.bat) is unchanged.
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo ERROR: Python environment ".venv" not found.
  echo Ask for help to finish setup first.
  echo.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -c "import sounddevice" 1>nul 2>nul
if errorlevel 1 (
  echo Installing microphone package: sounddevice ...
  "%~dp0.venv\Scripts\python.exe" -m pip install sounddevice
  if errorlevel 1 (
    echo ERROR: Could not install sounddevice.
    pause
    exit /b 1
  )
)

"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe_live.py" %*
set ERR=%ERRORLEVEL%

echo.
if %ERR% neq 0 (
  echo Finished with an error code: %ERR%
  echo The window will stay open so you can read the message.
  pause
  exit /b %ERR%
)

echo When finished, open live_output\Transcript_live.txt
echo Optional: copy the live_recording_*.wav into input\ and use run.bat
echo.
pause
endlocal
