@echo off
setlocal
cd /d "%~dp0"

echo.
echo Carmel Meeting Transcription (local)
echo ------------------------------------
echo Put ONE recording in the "input" folder, then run this again if needed.
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo ERROR: Python environment ".venv" not found.
  echo Ask for help to finish Prompt A setup first.
  echo.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe_meeting.py" %*
set ERR=%ERRORLEVEL%

echo.
if %ERR% neq 0 (
  echo Finished with an error code: %ERR%
  echo The window will stay open so you can read the message.
  pause
  exit /b %ERR%
)

echo When finished successfully, open:
echo   output\Transcript.txt
echo   output\Meeting_Notes.md
echo.
pause
endlocal
