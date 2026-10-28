# Carmel Meeting Transcription (local)

A simple tool that turns meeting audio into readable files on this computer. It works offline after models are downloaded. No cloud transcription service.

There are **two modes**. The original batch mode is unchanged.

## A) Batch (after the meeting) — existing

1. Put **ONE** recording into the `input` folder.  
   Allowed types: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`

2. Start the tool:
   - **Windows:** double-click `run.bat`
   - **Mac / Linux:** open a terminal in this folder and type: `sh run.sh`

3. Wait until it finishes (longer meetings take more time on a laptop without a dedicated NVIDIA GPU).

4. Open these two files in the `output` folder:
   - `Transcript.txt` — original-language words with time stamps and estimated speakers  
   - `Meeting_Notes.md` — draft English + places for you to confirm decisions and actions

5. **Always check against the recording:**
   - Are the **words** right?
   - Are the **speaker labels** right? (they are estimates only)
   - Is the **English draft** good enough? (it is machine-made and may miss Cantonese/Mandarin details)

## B) Live (during the meeting) — new

Near-real-time: the computer listens to the **microphone**, automatically splits on short silence (or about every 3–8 seconds), and scrolls text in the window. **You do not cut audio yourself.**

1. Start live mode:
   - **Windows:** double-click `run_live.bat`
   - **Mac / Linux:** `sh run_live.sh`

2. If asked, pick a microphone device number.

3. Speak. After a short pause, lines appear in the terminal (typical delay a few seconds).

4. Press **Ctrl+C** to stop.

5. Each live session writes a **matched pair** in `live_output/` (same timestamp):
   - `Transcript_live_YYYYMMDD_HHMMSS.txt`
   - `live_recording_YYYYMMDD_HHMMSS.wav`

6. **Optional (recommended after the meeting):** copy that `.wav` into `input/`, then use `run.bat` / `run.sh` for speaker labels + English draft + meeting notes.

Live mode defaults to **SPEED** (faster). For higher accuracy (slower): `run_live.bat --quality` or `sh run_live.sh --quality`.  
List mics: `run_live.bat --list-devices`

## Notes

- Future recordings (batch): same steps — replace the file in `input`, run again, open the two outputs. You do **not** edit any Python code.
- Optional faster batch run (less accurate): `run.bat --speed` or `sh run.sh --speed`
- Decisions and action items are **not** filled in automatically — please write them yourself after review.
- Live and batch use separate folders (`live_output/` vs `output/`) so a batch run does not delete the live transcript.
