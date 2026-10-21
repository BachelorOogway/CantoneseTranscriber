# Carmel Meeting Transcription (local)

A simple tool that turns **one** meeting recording into two readable files on this computer. It works offline after models are downloaded. No cloud transcription service.

## How to use (every time)

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

## Notes

- Future recordings: same steps — replace the file in `input`, run again, open the two outputs. You do **not** edit any Python code.
- Optional faster run (less accurate): `run.bat --speed` or `sh run.sh --speed`
- Decisions and action items are **not** filled in automatically — please write them yourself after review.
