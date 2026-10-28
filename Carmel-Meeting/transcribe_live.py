#!/usr/bin/env python3
"""
Live (near-real-time) meeting transcription: mic → silence/time chunks → Whisper.

Keeps the existing batch tool (transcribe_meeting.py) unchanged.
Outputs go under live_output/ so batch runs do not wipe them.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from transcribe_meeting import (
    detect_hardware,
    fmt_hms,
    progress as _progress,
    resource_guard,
    select_and_load_model,
)


def progress(msg: str) -> None:
    """Windows consoles are often GBK; strip unencodable chars instead of crashing."""
    try:
        _progress(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        _progress(safe)


def safe_device_name(name: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return name.encode(enc, errors="replace").decode(enc, errors="replace")

ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "live_output"
SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 480 @ 16 kHz
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 mono


@dataclass
class LiveChunk:
    start_sec: float
    audio: np.ndarray  # float32 mono [-1, 1]


class UtteranceCollector:
    """
    Auto-split mic audio: cut on silence, or force-cut at max_sec.
    No manual cutting.
    """

    def __init__(
        self,
        *,
        silence_ms: int = 700,
        max_sec: float = 8.0,
        min_speech_sec: float = 0.8,
        vad_aggressiveness: int = 2,
        energy_fallback_rms: float = 0.015,
    ) -> None:
        import webrtcvad

        self.vad = webrtcvad.Vad(int(np.clip(vad_aggressiveness, 0, 3)))
        self.silence_frames_needed = max(1, int(round(silence_ms / FRAME_MS)))
        self.max_frames = max(1, int(round(max_sec * 1000 / FRAME_MS)))
        self.min_speech_frames = max(1, int(round(min_speech_sec * 1000 / FRAME_MS)))
        self.energy_fallback_rms = energy_fallback_rms

        self._pcm_buf = bytearray()
        self._speech_frames: list[bytes] = []
        self._trailing_silence = 0
        self._in_speech = False
        self._session_frame = 0
        self._utterance_start_frame = 0

    def _is_speech(self, frame: bytes) -> bool:
        try:
            if self.vad.is_speech(frame, SR):
                return True
        except Exception:
            pass
        # Fallback if VAD rejects odd frames: simple RMS
        arr = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr**2))) if len(arr) else 0.0
        return rms >= self.energy_fallback_rms

    def feed(self, pcm_int16: bytes) -> list[LiveChunk]:
        """Accept raw int16 mono bytes; return zero or more finished chunks."""
        self._pcm_buf.extend(pcm_int16)
        out: list[LiveChunk] = []
        while len(self._pcm_buf) >= BYTES_PER_FRAME:
            frame = bytes(self._pcm_buf[:BYTES_PER_FRAME])
            del self._pcm_buf[:BYTES_PER_FRAME]
            chunk = self._feed_frame(frame)
            if chunk is not None:
                out.append(chunk)
            self._session_frame += 1
        return out

    def flush(self) -> Optional[LiveChunk]:
        if not self._in_speech or len(self._speech_frames) < self.min_speech_frames:
            self._speech_frames.clear()
            self._in_speech = False
            self._trailing_silence = 0
            return None
        return self._emit()

    def _feed_frame(self, frame: bytes) -> Optional[LiveChunk]:
        speech = self._is_speech(frame)
        if speech:
            if not self._in_speech:
                self._in_speech = True
                self._utterance_start_frame = self._session_frame
                self._speech_frames = []
            self._speech_frames.append(frame)
            self._trailing_silence = 0
            if len(self._speech_frames) >= self.max_frames:
                return self._emit()
            return None

        if not self._in_speech:
            return None

        self._speech_frames.append(frame)
        self._trailing_silence += 1
        if self._trailing_silence >= self.silence_frames_needed:
            if len(self._speech_frames) - self._trailing_silence >= self.min_speech_frames:
                return self._emit()
            # Too short — discard
            self._speech_frames.clear()
            self._in_speech = False
            self._trailing_silence = 0
        elif len(self._speech_frames) >= self.max_frames:
            return self._emit()
        return None

    def _emit(self) -> LiveChunk:
        # Drop trailing silence frames from the chunk (keep a little)
        keep_silence = min(self._trailing_silence, 3)
        end = len(self._speech_frames) - self._trailing_silence + keep_silence
        end = max(end, 0)
        frames = self._speech_frames[:end] if end else list(self._speech_frames)
        start_sec = self._utterance_start_frame * FRAME_MS / 1000.0
        pcm = b"".join(frames)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._speech_frames.clear()
        self._in_speech = False
        self._trailing_silence = 0
        return LiveChunk(start_sec=start_sec, audio=audio)


def list_input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    devices = sd.query_devices()
    out: list[tuple[int, str]] = []
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) > 0:
            name = safe_device_name(str(d.get("name", f"Device {i}")))
            out.append((i, name))
    return out


def pick_device(preferred: Optional[int]) -> int:
    import sounddevice as sd

    devices = list_input_devices()
    if not devices:
        raise SystemExit(
            "No microphone input device found.\n"
            "Plug in a mic / allow microphone access, then run again."
        )
    if preferred is not None:
        ids = {i for i, _ in devices}
        if preferred not in ids:
            progress(f"Device id {preferred} is not a valid input. Available:")
            for i, name in devices:
                progress(f"  [{i}] {name}")
            raise SystemExit(1)
        return preferred

    if len(devices) == 1:
        return devices[0][0]

    progress("Available microphones:")
    for i, name in devices:
        progress(f"  [{i}] {name}")
    try:
        default = sd.default.device[0]
    except Exception:
        default = devices[0][0]
    raw = input(f"Choose device id [{default}]: ").strip()
    if not raw:
        return int(default) if default in {i for i, _ in devices} else devices[0][0]
    try:
        choice = int(raw)
    except ValueError:
        raise SystemExit("Please enter a numeric device id.")
    if choice not in {i for i, _ in devices}:
        raise SystemExit(f"Invalid device id: {choice}")
    return choice


def open_wav_writer(path: Path) -> wave.Wave_write:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    return w


def append_transcript_line(path: Path, start: float, end: float, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{fmt_hms(start)} - {fmt_hms(end)}] {text}\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def transcribe_chunk(model: Any, audio: np.ndarray) -> str:
    segments_iter, _info = model.transcribe(
        audio,
        task="transcribe",
        language=None,
        vad_filter=False,  # already silence-split
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    parts: list[str] = []
    for s in segments_iter:
        t = (s.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def run_live(
    *,
    mode: str,
    model_override: Optional[str],
    device_id: Optional[int],
    max_sec: float,
    silence_ms: int,
    list_devices_only: bool,
) -> int:
    if list_devices_only:
        for i, name in list_input_devices():
            progress(f"[{i}] {name}")
        return 0

    try:
        import sounddevice as sd
    except ImportError:
        progress(
            "Missing package: sounddevice.\n"
            "In Carmel-Meeting folder run:\n"
            "  .venv\\Scripts\\python.exe -m pip install sounddevice\n"
            "Then run live mode again."
        )
        return 1

    hw = detect_hardware()
    progress(
        f"Computer: {hw.os_name} | CPU x{hw.cpu_logical} | "
        f"RAM {hw.ram_total_gb:.1f} GB free≈{hw.ram_available_gb:.1f} | "
        f"NVIDIA: {'yes' if hw.nvidia_gpu else 'no'}"
    )
    block = resource_guard(hw)
    if block:
        progress(block)
        return 2

    progress("Loading speech model (first time may download)...")
    try:
        model, choice = select_and_load_model(hw, mode, model_override)
    except SystemExit as e:
        progress(str(e))
        return 3
    progress(
        f"Using model '{choice.name}' on {choice.device} ({choice.compute_type}), mode={choice.mode}."
    )

    mic = pick_device(device_id)
    dev_name = sd.query_devices(mic)["name"]
    progress(f"Microphone: [{mic}] {dev_name}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    # One session → one pair sharing the same stamp
    wav_path = LIVE_DIR / f"live_recording_{stamp}.wav"
    transcript_path = LIVE_DIR / f"Transcript_live_{stamp}.txt"
    transcript_path.write_text(
        "LIVE TRANSCRIPT (auto-split; not speaker-labeled)\n"
        f"Session id: {stamp}\n"
        f"Started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Model: {choice.name} / {choice.device} / {choice.mode}\n"
        f"Paired recording: {wav_path.name}\n"
        "Press Ctrl+C to stop.\n"
        "----\n",
        encoding="utf-8",
        newline="\n",
    )

    chunk_q: queue.Queue[Optional[LiveChunk]] = queue.Queue(maxsize=32)
    pcm_q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=256)
    stop = threading.Event()
    stats = {"chunks": 0, "chars": 0}
    err_box: deque[str] = deque(maxlen=5)

    collector = UtteranceCollector(silence_ms=silence_ms, max_sec=max_sec)
    wav_lock = threading.Lock()
    wav_writer = open_wav_writer(wav_path)

    def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            err_box.append(str(status))
        # indata: float32 shape (frames, channels)
        mono = indata[:, 0] if indata.ndim > 1 else indata.reshape(-1)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        try:
            pcm_q.put_nowait(pcm)
        except queue.Full:
            pass

    def feeder_loop() -> None:
        while not stop.is_set():
            try:
                pcm = pcm_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if pcm is None:
                break
            with wav_lock:
                try:
                    wav_writer.writeframes(pcm)
                except Exception as e:
                    err_box.append(f"wav write: {e}")
            for ch in collector.feed(pcm):
                try:
                    chunk_q.put(ch, timeout=1.0)
                except queue.Full:
                    err_box.append("chunk queue full; dropping a segment")

    def worker_loop() -> None:
        while not stop.is_set():
            try:
                ch = chunk_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if ch is None:
                break
            if ch.audio.size < int(0.4 * SR):
                continue
            try:
                text = transcribe_chunk(model, ch.audio)
            except Exception as e:
                err_box.append(f"asr: {type(e).__name__}: {e}")
                continue
            if not text:
                continue
            dur = len(ch.audio) / float(SR)
            end = ch.start_sec + dur
            line = f"[{fmt_hms(ch.start_sec)} - {fmt_hms(end)}] {text}"
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(safe_device_name(line), flush=True)
            append_transcript_line(transcript_path, ch.start_sec, end, text)
            stats["chunks"] += 1
            stats["chars"] += len(text)

    feeder = threading.Thread(target=feeder_loop, name="live-feeder", daemon=True)
    worker = threading.Thread(target=worker_loop, name="live-asr", daemon=True)
    feeder.start()
    worker.start()

    progress("")
    progress("Live mode started. Speak into the mic — text appears after a short pause.")
    progress(f"Writing: {transcript_path}")
    progress(f"Recording: {wav_path}")
    progress("After the meeting: copy the .wav into input\\ and use run.bat for speakers/English notes.")
    progress("Press Ctrl+C to stop.")
    progress("")

    stream = None
    try:
        stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=mic,
            blocksize=FRAME_SAMPLES,
            callback=audio_callback,
        )
        stream.start()
        while not stop.is_set():
            time.sleep(0.5)
            if err_box:
                while err_box:
                    progress(f"Note: {err_box.popleft()}")
    except KeyboardInterrupt:
        progress("\nStopping...")
    except Exception as e:
        progress(f"Microphone error: {type(e).__name__}: {e}")
        stop.set()
        return 1
    finally:
        stop.set()
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        # Drain remaining pcm then flush utterance
        time.sleep(0.3)
        while True:
            try:
                pcm = pcm_q.get_nowait()
            except queue.Empty:
                break
            if pcm:
                with wav_lock:
                    try:
                        wav_writer.writeframes(pcm)
                    except Exception:
                        pass
                for ch in collector.feed(pcm):
                    try:
                        chunk_q.put_nowait(ch)
                    except queue.Full:
                        pass
        leftover = collector.flush()
        if leftover is not None:
            try:
                chunk_q.put_nowait(leftover)
            except queue.Full:
                pass
        try:
            chunk_q.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=120.0)
        feeder.join(timeout=5.0)
        with wav_lock:
            try:
                wav_writer.close()
            except Exception:
                pass

    with open(transcript_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("----\n")
        f.write(f"Stopped: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Segments: {stats['chunks']}\n")

    progress("")
    progress("Live session finished.")
    progress(f"1) {transcript_path}")
    progress(f"2) {wav_path}")
    progress("Optional: put the wav into input\\ and run run.bat for full batch (speakers + English notes).")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live mic transcription (auto silence/time split). Batch tool unchanged."
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use QUALITY model tier (slower). Default for live is SPEED.",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="SPEED mode (default for live; kept for symmetry with batch).",
    )
    parser.add_argument(
        "--model",
        choices=("base", "small", "medium", "large-v3"),
        default=None,
        help="Force Whisper model name.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone device id (see --list-devices).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone devices and exit.",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=8.0,
        help="Force-cut a segment after this many seconds of speech (default 8).",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=700,
        help="Silence length that ends a segment (default 700 ms).",
    )
    args = parser.parse_args(argv)

    mode = "QUALITY" if args.quality else "SPEED"
    if args.speed and args.quality:
        progress("Both --speed and --quality set; using QUALITY.")
        mode = "QUALITY"

    return run_live(
        mode=mode,
        model_override=args.model,
        device_id=args.device,
        max_sec=max(2.0, float(args.max_sec)),
        silence_ms=max(200, int(args.silence_ms)),
        list_devices_only=args.list_devices,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        raise SystemExit(130)
