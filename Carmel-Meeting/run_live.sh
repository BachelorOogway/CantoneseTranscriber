#!/bin/sh
# Usage (no chmod needed):  sh run_live.sh
cd "$(dirname "$0")" || exit 1

echo ""
echo "Carmel Meeting LIVE Transcription (local mic)"
echo "---------------------------------------------"
echo "Speak into the mic; text appears after short pauses."
echo "Existing batch tool (run.sh) is unchanged."
echo ""

if [ ! -x "./.venv/bin/python" ] && [ ! -f "./.venv/bin/python" ]; then
  echo "ERROR: Python environment .venv not found."
  echo "Ask for help to finish setup first."
  exit 1
fi

if ! ./.venv/bin/python -c "import sounddevice" 2>/dev/null; then
  echo "Installing microphone package: sounddevice ..."
  ./.venv/bin/python -m pip install sounddevice || {
    echo "ERROR: Could not install sounddevice."
    exit 1
  }
fi

./.venv/bin/python ./transcribe_live.py "$@"
ERR=$?

echo ""
if [ "$ERR" -ne 0 ]; then
  echo "Finished with an error code: $ERR"
  exit "$ERR"
fi

echo "When finished, open the matching pair in live_output/:"
echo "  Transcript_live_YYYYMMDD_HHMMSS.txt"
echo "  live_recording_YYYYMMDD_HHMMSS.wav"
echo "Optional: copy that wav into input/ and use run.sh"
echo ""
exit 0
