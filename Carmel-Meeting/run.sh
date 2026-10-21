#!/bin/sh
# Usage (no chmod needed):  sh run.sh
cd "$(dirname "$0")" || exit 1

echo ""
echo "Carmel Meeting Transcription (local)"
echo "------------------------------------"
echo "Put ONE recording in the input folder, then run again if needed."
echo ""

if [ ! -x "./.venv/bin/python" ] && [ ! -f "./.venv/bin/python" ]; then
  echo "ERROR: Python environment .venv not found."
  echo "Ask for help to finish Prompt A setup first."
  exit 1
fi

./.venv/bin/python ./transcribe_meeting.py "$@"
ERR=$?

echo ""
if [ "$ERR" -ne 0 ]; then
  echo "Finished with an error code: $ERR"
  exit "$ERR"
fi

echo "When finished successfully, open:"
echo "  output/Transcript.txt"
echo "  output/Meeting_Notes.md"
echo ""
exit 0
