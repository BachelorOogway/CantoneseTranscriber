#!/usr/bin/env python3
"""
Local-first meeting transcription for school teachers.
Cantonese / Mandarin / English (incl. code-switching). No cloud APIs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
TRANSCRIPT_NAME = "Transcript.txt"
NOTES_NAME = "Meeting_Notes.md"
SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mov"}

# Whisper: public ungated multilingual models only
ALLOWED_MODELS = ("base", "small", "medium", "large-v3")

# Speaker clustering (conservative cosine distance threshold for agglomerative)
SPEAKER_DISTANCE_THRESHOLD = 0.32
WINDOW_SEC = 2.5
WINDOW_HOP_SEC = 1.25
MIN_WINDOW_SEC = 1.2
MIN_WINDOW_RMS = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_hms(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def progress(msg: str) -> None:
    print(msg, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Hardware / resources
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    os_name: str
    arch: str
    cpu_logical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    nvidia_gpu: bool
    vram_gb: Optional[float]
    cuda_usable: bool  # verified only after tiny inference
    notes: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareInfo:
    import platform

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    nvidia = False
    vram_gb = None
    notes: list[str] = []

    # Detect NVIDIA presence only; do NOT assume CUDA works
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            nvidia = True
            try:
                vram_gb = float(r.stdout.strip().splitlines()[0].strip()) / 1024.0
            except ValueError:
                vram_gb = None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        notes.append("No nvidia-smi; treating as CPU-only.")

    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=vm.total / (1024**3),
        ram_available_gb=vm.available / (1024**3),
        disk_free_gb=disk.free / (1024**3),
        nvidia_gpu=nvidia,
        vram_gb=vram_gb,
        cuda_usable=False,
        notes=notes,
    )


def resource_guard(hw: HardwareInfo) -> Optional[str]:
    """Pilot thresholds: stop if clearly unsafe."""
    if hw.ram_total_gb < 8:
        return (
            f"Installed RAM is about {hw.ram_total_gb:.1f} GB (below the 8 GB pilot minimum). "
            "Please free memory or use a computer with more RAM, then try again."
        )
    if hw.ram_available_gb < 1.5:
        return (
            f"Available memory is only about {hw.ram_available_gb:.1f} GB (dangerously low). "
            "Close other programs and try again."
        )
    if hw.disk_free_gb < 15:
        return (
            f"Free disk space is about {hw.disk_free_gb:.1f} GB (below the 15 GB pilot minimum). "
            "Free some disk space and try again."
        )
    return None


def cpu_quality_model(ram_total_gb: float) -> str:
    if ram_total_gb >= 32:
        return "medium"
    if ram_total_gb >= 16:
        return "small"
    return "base"


def gpu_quality_model(vram_gb: float) -> str:
    if vram_gb >= 8:
        return "large-v3"
    if vram_gb >= 6:
        return "medium"
    if vram_gb >= 4:
        return "small"
    return "small"


def step_down(model: str) -> Optional[str]:
    order = ["large-v3", "medium", "small", "base"]
    if model not in order:
        return "base"
    i = order.index(model)
    if i >= len(order) - 1:
        return None
    return order[i + 1]


def speed_tier(model: str, device: str) -> str:
    """One tier lower; CPU min base, GPU min small."""
    down = step_down(model) or model
    if device == "cuda" and down == "base":
        return "small"
    return down


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def list_input_recordings() -> list[Path]:
    if not INPUT_DIR.is_dir():
        return []
    files = []
    for p in sorted(INPUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Audio decode (PyAV) → mono float32 16 kHz
# ---------------------------------------------------------------------------

def decode_audio_mono_16k(path: Path, max_seconds: Optional[float] = None) -> tuple[np.ndarray, float]:
    """
    Decode audio track with PyAV (bundled FFmpeg). Returns (samples float32 [-1,1], duration_sec).
    Does not modify the source file.
    """
    import av

    samples: list[np.ndarray] = []
    duration = 0.0
    try:
        container = av.open(str(path))
    except Exception as e:
        raise RuntimeError(
            f"Could not read this file as audio/video: {path.name}\n"
            f"Technical detail: {type(e).__name__}: {e}"
        ) from e

    try:
        if not container.streams.audio:
            raise RuntimeError(f"No audio track found in: {path.name}")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
        collected = 0.0
        for frame in container.decode(stream):
            frame.pts = None  # avoid resample timestamp issues
            out = resampler.resample(frame)
            if out is None:
                continue
            frames = out if isinstance(out, list) else [out]
            for f in frames:
                arr = f.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                samples.append(arr)
                collected += len(arr) / 16000.0
                if max_seconds is not None and collected >= max_seconds:
                    break
            if max_seconds is not None and collected >= max_seconds:
                break
        # flush
        out = resampler.resample(None)
        if out is not None:
            frames = out if isinstance(out, list) else [out]
            for f in frames:
                arr = f.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                samples.append(np.asarray(arr, dtype=np.float32).reshape(-1))

        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
    finally:
        container.close()

    if not samples:
        raise RuntimeError(f"No audio samples decoded from: {path.name}")

    audio = np.concatenate(samples)
    if max_seconds is not None:
        n = int(max_seconds * 16000)
        audio = audio[:n]
        duration = min(duration if duration > 0 else len(audio) / 16000.0, max_seconds)
    else:
        duration = duration if duration > 0 else len(audio) / 16000.0

    # clip
    audio = np.clip(audio, -1.0, 1.0)
    return audio, float(duration)


# ---------------------------------------------------------------------------
# Whisper load + tiny inference probe
# ---------------------------------------------------------------------------

@dataclass
class ModelChoice:
    name: str
    device: str
    compute_type: str
    mode: str  # QUALITY or SPEED
    warnings: list[str] = field(default_factory=list)


def tiny_synthetic_audio(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    # quiet tone - for load/inference smoke test only, NOT speech quality
    return (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def try_load_and_infer(model_name: str, device: str, compute_type: str) -> tuple[Any, Optional[str]]:
    """Return (model, None) on success, or (None, error_message)."""
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        return None, f"Model load failed ({model_name}/{device}/{compute_type}): {type(e).__name__}: {e}"

    audio = tiny_synthetic_audio(1.0)
    try:
        segments, info = model.transcribe(
            audio,
            language=None,
            task="transcribe",
            vad_filter=False,
            beam_size=1,
        )
        # Materialise iterator so inference actually runs
        _ = list(segments)
        _ = info
    except Exception as e:
        del model
        return None, f"Inference test failed ({model_name}/{device}/{compute_type}): {type(e).__name__}: {e}"
    return model, None


def select_and_load_model(
    hw: HardwareInfo,
    mode: str,
    override: Optional[str],
) -> tuple[Any, ModelChoice]:
    warnings: list[str] = []

    if override:
        if override not in ALLOWED_MODELS:
            raise SystemExit(
                f"Unsupported model '{override}'. Allowed: {', '.join(ALLOWED_MODELS)}"
            )
        candidates_cpu = [override]
        candidates_gpu = [override]
    else:
        cpu_m = cpu_quality_model(hw.ram_total_gb)
        # Prefer safer model if available RAM is tight relative to tier
        if hw.ram_available_gb < 4 and cpu_m == "medium":
            warnings.append(
                f"Available RAM is only ~{hw.ram_available_gb:.1f} GB; starting with 'small' instead of 'medium'."
            )
            cpu_m = "small"
        if hw.ram_available_gb < 2.5 and cpu_m == "small":
            warnings.append(
                f"Available RAM is only ~{hw.ram_available_gb:.1f} GB; starting with 'base'."
            )
            cpu_m = "base"

        if mode.upper() == "SPEED":
            cpu_m = speed_tier(cpu_m, "cpu")
            gpu_m = speed_tier(
                gpu_quality_model(hw.vram_gb or 0) if hw.nvidia_gpu else "small", "cuda"
            )
        else:
            gpu_m = gpu_quality_model(hw.vram_gb or 0) if hw.nvidia_gpu else "small"

        candidates_cpu = []
        m = cpu_m
        while m:
            candidates_cpu.append(m)
            m = step_down(m)

        candidates_gpu = []
        m = gpu_m
        while m:
            candidates_gpu.append(m)
            m = step_down(m)
            if m == "base":
                break  # GPU minimum small in profiles; still allow fallthrough to CPU

    # macOS / no NVIDIA → CPU only for this MVP
    is_macos = sys.platform == "darwin"
    try_cuda = hw.nvidia_gpu and not is_macos

    if try_cuda:
        for name in candidates_gpu:
            for ctype in ("int8_float16", "float16", "int8"):
                progress(f"Trying GPU model '{name}' ({ctype})...")
                model, err = try_load_and_infer(name, "cuda", ctype)
                if model is not None:
                    hw.cuda_usable = True
                    return model, ModelChoice(name, "cuda", ctype, mode.upper(), warnings)
                warnings.append(err or "GPU attempt failed")
        warnings.append("CUDA not usable for inference; falling back to CPU.")

    for name in candidates_cpu:
        progress(f"Trying CPU model '{name}' (int8)...")
        model, err = try_load_and_infer(name, "cpu", "int8")
        if model is not None:
            return model, ModelChoice(name, "cpu", "int8", mode.upper(), warnings)
        warnings.append(err or "CPU attempt failed")
        # Download blocked?
        joined = " ".join(warnings).lower()
        if any(x in joined for x in ("401", "403", "gated", "huggingface", "connection", "timed out", "resolve")):
            raise SystemExit(
                "Could not download or load a Whisper model.\n"
                "This tool needs internet once to download a public model (no login).\n"
                "If school network blocks the download, ask school IT to allow access to "
                "Hugging Face model downloads for faster-whisper / CTranslate2, then try again.\n"
                f"Last error: {err}"
            )

    raise SystemExit(
        "Could not load any Whisper model on this computer.\n"
        + "\n".join(warnings[-5:])
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    language_hint: Optional[str] = None


def run_transcribe(model: Any, audio: np.ndarray) -> tuple[list[Segment], dict[str, Any], list[str]]:
    warnings: list[str] = []
    # Multilingual: do not force one language for the whole file
    segments_iter, info = model.transcribe(
        audio,
        task="transcribe",
        language=None,  # auto as starting point only
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    segs: list[Segment] = []
    for s in segments_iter:  # materialise - inference must run
        text = (s.text or "").strip()
        if not text:
            continue
        segs.append(Segment(float(s.start), float(s.end), text))

    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)
    meta = {"language": lang, "language_probability": prob}
    if lang and prob is not None and prob < 0.6:
        warnings.append(
            f"Automatic language guess '{lang}' is uncertain (confidence ~{prob:.2f}). "
            "Treating language as mixed / not verified."
        )
    if not segs:
        warnings.append("No speech text was produced (silence, noise, or model found nothing).")
    return segs, meta, warnings


def run_translate(model: Any, audio: np.ndarray) -> tuple[list[Segment], list[str], bool]:
    """
    Second pass: task=translate → English draft.
    Returns (segments, warnings, reliable_enough).
    """
    warnings: list[str] = []
    try:
        segments_iter, info = model.transcribe(
            audio,
            task="translate",
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            condition_on_previous_text=True,
        )
        segs: list[Segment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            segs.append(Segment(float(s.start), float(s.end), text))
    except Exception as e:
        return [], [f"English translation pass failed: {type(e).__name__}: {e}"], False

    if not segs:
        return [], ["English translation produced no text."], False

    # Heuristic quality gate: if most text looks like CJK, treat as unreliable
    joined = " ".join(s.text for s in segs)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    letters = len(re.findall(r"[A-Za-z]", joined))
    if cjk > letters * 2 and cjk > 20:
        warnings.append(
            "Translation output still looks mostly Chinese; not treating it as reliable English."
        )
        return segs, warnings, False
    if len(joined.strip()) < 8:
        return segs, ["English translation too short to trust."], False

    return segs, warnings, True


# ---------------------------------------------------------------------------
# Speaker embeddings (Resemblyzer) + clustering
# ---------------------------------------------------------------------------

def energy_windows(audio: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
    """Simple energy-based speech windows (~1.5-4 s)."""
    win = int(WINDOW_SEC * sr)
    hop = int(WINDOW_HOP_SEC * sr)
    min_len = int(MIN_WINDOW_SEC * sr)
    windows: list[tuple[float, float]] = []
    if len(audio) < min_len:
        return windows
    for start in range(0, max(1, len(audio) - min_len + 1), hop):
        end = min(start + win, len(audio))
        chunk = audio[start:end]
        if len(chunk) < min_len:
            break
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < MIN_WINDOW_RMS:
            continue
        windows.append((start / sr, end / sr))
        if end >= len(audio):
            break
    return windows


def assign_speakers(
    audio: np.ndarray,
    segments: list[Segment],
) -> tuple[list[str], list[str], bool]:
    """
    Returns (label_per_segment, warnings, succeeded).
    Labels like 'Speaker 1' or 'Speaker ?' / 'Unverified'.
    """
    warnings: list[str] = []
    n = len(segments)
    if n == 0:
        return [], warnings, True

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering
    except Exception as e:
        warnings.append(f"Speaker tools unavailable: {type(e).__name__}: {e}")
        return ["Unverified"] * n, warnings, False

    try:
        encoder = VoiceEncoder()
        wins = energy_windows(audio)
        # Also add windows from Whisper segment midpoints if few energy windows
        if len(wins) < 2:
            for seg in segments:
                mid = 0.5 * (seg.start + seg.end)
                half = WINDOW_SEC / 2
                a = max(0.0, mid - half)
                b = min(len(audio) / 16000.0, mid + half)
                if b - a >= MIN_WINDOW_SEC:
                    wins.append((a, b))

        if len(wins) < 1:
            warnings.append("Not enough clear speech windows for speaker labels.")
            return ["Unverified"] * n, warnings, False

        embeds = []
        win_times = []
        for a, b in wins:
            i0, i1 = int(a * 16000), int(b * 16000)
            chunk = audio[i0:i1]
            if len(chunk) < int(MIN_WINDOW_SEC * 16000):
                continue
            # resemblyzer expects float wav; preprocess_wav accepts path or ndarray
            wav = preprocess_wav(chunk, source_sr=16000)
            if len(wav) < 1600:
                continue
            emb = encoder.embed_utterance(wav)
            embeds.append(emb)
            win_times.append((a, b))

        if len(embeds) < 1:
            warnings.append("Could not compute voice embeddings.")
            return ["Unverified"] * n, warnings, False

        X = np.vstack(embeds)
        # L2 normalize for cosine
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        if len(embeds) == 1:
            cluster_ids = np.array([0])
        else:
            # cosine distance = 1 - cosine similarity; average linkage
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=SPEAKER_DISTANCE_THRESHOLD,
            )
            cluster_ids = clustering.fit_predict(X)

        # Map cluster → Speaker N by first appearance in time
        order: list[int] = []
        for cid, (a, _) in sorted(zip(cluster_ids.tolist(), win_times), key=lambda t: t[1][0]):
            if cid not in order:
                order.append(cid)
        cid_to_label = {cid: f"Speaker {i+1}" for i, cid in enumerate(order)}

        # Assign each transcript segment by majority overlap of windows
        labels: list[str] = []
        for seg in segments:
            votes: dict[int, float] = {}
            for cid, (a, b) in zip(cluster_ids.tolist(), win_times):
                overlap = max(0.0, min(seg.end, b) - max(seg.start, a))
                if overlap > 0:
                    votes[cid] = votes.get(cid, 0.0) + overlap
            if not votes:
                labels.append("Speaker ?")
                continue
            best = max(votes, key=votes.get)
            total = sum(votes.values())
            if votes[best] / total < 0.55:
                labels.append("Speaker ?")
            else:
                labels.append(cid_to_label.get(best, "Speaker ?"))

        # Mark very short turns uncertain
        for i, seg in enumerate(segments):
            if seg.end - seg.start < 0.6 and labels[i].startswith("Speaker "):
                labels[i] = labels[i]  # keep but note
        short = sum(1 for s in segments if s.end - s.start < 0.6)
        if short:
            warnings.append(
                f"{short} very short turn(s) may have uncertain speaker labels."
            )

        return labels, warnings, True
    except Exception as e:
        warnings.append(f"Speaker labeling failed: {type(e).__name__}: {e}")
        return ["Unverified"] * n, warnings, False


def maybe_split_on_speaker_change(
    segments: list[Segment],
    labels: list[str],
) -> tuple[list[Segment], list[str]]:
    """
    MVP: do not invent splits without strong evidence.
    Keep segments as-is; splitting only when adjacent windows disagree is deferred
    to label Speaker ? rather than false certainty (already handled above).
    """
    return segments, labels


# ---------------------------------------------------------------------------
# Language line for transcript header
# ---------------------------------------------------------------------------

def language_header(meta: dict[str, Any], segments: list[Segment], warn: list[str]) -> str:
    lang = meta.get("language")
    prob = meta.get("language_probability")
    if not lang or (prob is not None and prob < 0.6):
        return "mixed / not verified"
    # Whisper often reports 'zh' for both Cantonese and Mandarin
    if lang in ("zh", "yue", "zh-cn", "zh-tw"):
        return "Chinese (Cantonese/Mandarin not distinguished) / not fully verified"
    if lang == "en":
        # still may be mixed
        has_cjk = any(re.search(r"[\u4e00-\u9fff]", s.text) for s in segments)
        if has_cjk:
            return "mixed / not verified"
        return "English (auto-detect; please verify)"
    return "mixed / not verified"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_transcript(
    path: Path,
    filename: str,
    duration: float,
    language_line: str,
    segments: list[Segment],
    labels: list[str],
) -> None:
    lines = [
        "MEETING TRANSCRIPT",
        f"File: {filename}",
        f"Duration: {fmt_hms(duration)}",
        f"Language: {language_line}",
        "Note: Chinese speech may include Cantonese and Mandarin.",
        "Speaker labels are estimates; please check against the recording.",
        "",
    ]
    for seg, lab in zip(segments, labels):
        lines.append(f"[{fmt_hms(seg.start)} - {fmt_hms(seg.end)}] {lab}:")
        lines.append(seg.text)
        lines.append("")
    if not segments:
        lines.append("[00:00:00 - " + fmt_hms(duration) + "] Unverified:")
        lines.append("(No speech text detected.)")
        lines.append("")
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def write_meeting_notes(
    path: Path,
    translate_segs: list[Segment],
    translate_labels: Optional[list[str]],
    translation_ok: bool,
    all_warnings: list[str],
    model_choice: ModelChoice,
    speaker_ok: bool,
) -> None:
    warn_lines = list(all_warnings)
    warn_lines.append(
        f"Model used: {model_choice.name} on {model_choice.device} "
        f"({model_choice.compute_type}), mode={model_choice.mode}."
    )
    if not speaker_ok:
        warn_lines.append("Speaker labels were not verified / fell back to Unverified.")
    if not translation_ok:
        warn_lines.append("English translation was skipped or not reliable.")

    # ENGLISH TRANSCRIPT section
    if translation_ok and translate_segs:
        eng_parts = [
            "Machine-generated approximate English (Whisper translate). "
            "May be incomplete for mixed Cantonese/Mandarin/English. Please check the recording.",
            "",
        ]
        for i, seg in enumerate(translate_segs):
            lab = ""
            if translate_labels and i < len(translate_labels):
                tlab = translate_labels[i]
                if tlab.startswith("Speaker ") and "?" not in tlab and tlab != "Unverified":
                    lab = f" {tlab}"
            eng_parts.append(f"[{fmt_hms(seg.start)} - {fmt_hms(seg.end)}]{lab}")
            eng_parts.append(seg.text)
            eng_parts.append("")
        eng_body = "\n".join(eng_parts).rstrip()
    else:
        eng_body = (
            "English translation not reliably available; please review the original transcript"
        )

    text = "\n".join(
        [
            "# MEETING OVERVIEW",
            "To be confirmed by teacher",
            "",
            "# ENGLISH TRANSCRIPT (DRAFT)",
            eng_body,
            "",
            "# DECISIONS (TEACHER REVIEW)",
            "To be confirmed by teacher",
            "",
            "# ACTION ITEMS (TEACHER REVIEW)",
            "To be confirmed by teacher",
            "",
            "# WARNINGS",
            *(f"- {w}" for w in warn_lines),
            "",
        ]
    )
    atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# Timing estimate / teacher prompts
# ---------------------------------------------------------------------------

def estimate_and_confirm(duration: float, sample_seconds: float, sample_wall: float) -> bool:
    if sample_seconds <= 0 or sample_wall <= 0:
        return True
    rate = sample_wall / sample_seconds  # wall seconds per audio second
    est = rate * duration
    progress(
        f"Rough time estimate for the full recording: about {fmt_hms(est)} "
        f"(based on a short sample; actual time varies)."
    )
    if est > 20 * 60:  # > 20 minutes wall time
        ans = input("This may take a long time on this computer. Continue with the FULL recording? [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    if duration > 3 * 60:
        ans = input(
            "Recording is longer than 3 minutes. Process only the FIRST 3 minutes as a trial? "
            "Type 'trial' for 3 minutes, or 'full' for the whole file: "
        ).strip().lower()
        if ans in ("trial", "t", "3"):
            return "trial"  # type: ignore
        if ans in ("full", "f", "y", "yes"):
            return True
        progress("Cancelled.")
        return False
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    mode: str,
    model_override: Optional[str],
    max_seconds: Optional[float],
    skip_confirm: bool,
) -> int:
    hw = detect_hardware()
    progress(
        f"Computer: {hw.os_name} | CPU x{hw.cpu_logical} | "
        f"RAM {hw.ram_total_gb:.1f} GB total, {hw.ram_available_gb:.1f} GB free | "
        f"Disk {hw.disk_free_gb:.1f} GB free | "
        f"NVIDIA: {'yes' if hw.nvidia_gpu else 'no'}"
    )
    block = resource_guard(hw)
    if block:
        progress(block)
        return 2

    # Optional short timing probe before full decode of huge files - decode duration first
    progress(f"Reading audio from: {path.name} (original file will not be changed)...")
    try:
        # Quick duration via container
        import av

        with av.open(str(path)) as c:
            if c.duration is not None:
                full_duration = float(c.duration / av.time_base)
            elif c.streams.audio:
                st = c.streams.audio[0]
                full_duration = float(st.duration * st.time_base) if st.duration else 0.0
            else:
                full_duration = 0.0
    except Exception as e:
        progress(f"Could not open file: {type(e).__name__}: {e}")
        return 1

    limit = max_seconds
    if not skip_confirm and full_duration > 180 and max_seconds is None:
        ans = input(
            f"This recording looks about {fmt_hms(full_duration)}. "
            "Type 'trial' to process only the first 3 minutes, "
            "'full' for the whole recording, or 'n' to cancel: "
        ).strip().lower()
        if ans in ("n", "no", "cancel"):
            progress("Cancelled. No output written.")
            return 0
        if ans in ("trial", "t", "3"):
            limit = 180.0
            progress("Trial mode: first 3 minutes only.")
        elif ans not in ("full", "f", "y", "yes"):
            progress("Cancelled. No output written.")
            return 0

    try:
        audio, duration = decode_audio_mono_16k(path, max_seconds=limit)
    except RuntimeError as e:
        progress(str(e))
        return 1

    progress(f"Audio ready: {fmt_hms(duration)} (mono 16 kHz). Loading speech model...")
    try:
        model, choice = select_and_load_model(hw, mode, model_override)
    except SystemExit as e:
        progress(str(e))
        return 3

    progress(
        f"Using model '{choice.name}' on {choice.device} ({choice.compute_type}), mode={choice.mode}."
    )
    for w in choice.warnings:
        progress(f"Note: {w}")

    # Optional speed estimate on CPU for longer files
    all_warnings: list[str] = list(choice.warnings)
    if choice.device == "cpu" and duration > 120 and not skip_confirm:
        import time

        sample_len = min(20.0, duration)
        sample = audio[: int(sample_len * 16000)]
        progress("Running a short timing sample (not the final transcript)...")
        t0 = time.perf_counter()
        _segs, _meta, _w = run_transcribe(model, sample)
        wall = time.perf_counter() - t0
        est = (wall / max(sample_len, 0.1)) * duration
        progress(f"Estimated full processing time: about {fmt_hms(est)}.")
        if est > 15 * 60:
            ans = input("Continue with this recording? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                progress("Stopped before full transcription. No teacher output written.")
                return 0
            # offer SPEED
            if choice.mode == "QUALITY":
                ans2 = input("Switch to SPEED mode (faster, less accurate) for this run? [y/N]: ").strip().lower()
                if ans2 in ("y", "yes"):
                    del model
                    model, choice = select_and_load_model(hw, "SPEED", None)
                    progress(
                        f"Switched to '{choice.name}' on {choice.device} ({choice.compute_type})."
                    )

    progress("Transcribing (original languages; may take a while)...")
    segments, meta, tw = run_transcribe(model, audio)
    all_warnings.extend(tw)

    progress("Estimating speakers (best-effort; not verified identities)...")
    labels, sw, speaker_ok = assign_speakers(audio, segments)
    all_warnings.extend(sw)
    segments, labels = maybe_split_on_speaker_change(segments, labels)
    if not speaker_ok:
        labels = ["Unverified"] * len(segments)

    progress("Draft English translation pass (optional; may be imperfect)...")
    tsegs, trw, translation_ok = run_translate(model, audio)
    all_warnings.extend(trw)
    # Align speaker labels to translation by time overlap only when reliable
    tlabels: Optional[list[str]] = None
    if translation_ok and speaker_ok and segments and tsegs:
        tlabels = []
        for ts in tsegs:
            votes: dict[str, float] = {}
            for seg, lab in zip(segments, labels):
                if not lab.startswith("Speaker ") or "?" in lab:
                    continue
                overlap = max(0.0, min(ts.end, seg.end) - max(ts.start, seg.start))
                if overlap > 0:
                    votes[lab] = votes.get(lab, 0.0) + overlap
            if votes:
                tlabels.append(max(votes, key=votes.get))
            else:
                tlabels.append("")

    lang_line = language_header(meta, segments, all_warnings)

    out_t = OUTPUT_DIR / TRANSCRIPT_NAME
    out_n = OUTPUT_DIR / NOTES_NAME
    # Clear other files in output/ - only the two teacher-facing results should remain
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUTPUT_DIR.iterdir():
        if p.is_file() and p.name not in (TRANSCRIPT_NAME, NOTES_NAME):
            try:
                p.unlink()
            except OSError:
                pass

    write_transcript(out_t, path.name, duration, lang_line, segments, labels if labels else ["Unverified"] * len(segments))
    write_meeting_notes(out_n, tsegs, tlabels, translation_ok, all_warnings, choice, speaker_ok)

    # Ensure only those two files
    for p in list(OUTPUT_DIR.iterdir()):
        if p.is_file() and p.name not in (TRANSCRIPT_NAME, NOTES_NAME):
            try:
                p.unlink()
            except OSError:
                pass

    progress("")
    progress("Done.")
    progress(f"1) {out_t}")
    progress(f"2) {out_n}")
    progress("Please check: (1) words, (2) speaker labels, (3) English draft against the recording.")
    progress("Decisions and action items are NOT auto-filled - please confirm them yourself.")
    # Do not print transcript contents
    return 0


# ---------------------------------------------------------------------------
# Self-test (synthetic tone - NOT a speech-quality claim)
# ---------------------------------------------------------------------------

def write_synthetic_wav(path: Path, seconds: float = 2.0, sr: int = 16000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # 0.5 s silence + tone + silence - decode/format test only
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone[: int(0.3 * sr)] = 0
    tone[-int(0.3 * sr) :] = 0
    pcm = (np.clip(tone, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def run_self_test() -> int:
    progress("Running synthetic self-test (tone/silence - NOT a speech accuracy test)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carmel_selftest_") as td:
        wav_path = Path(td) / "synthetic_tone.wav"
        write_synthetic_wav(wav_path, 2.0)
        try:
            audio, dur = decode_audio_mono_16k(wav_path)
        except Exception as e:
            progress(f"SELF-TEST FAIL: decode: {e}")
            return 1
        if audio.ndim != 1 or abs(dur - 2.0) > 0.25:
            progress(f"SELF-TEST FAIL: unexpected audio shape/duration ({audio.shape}, {dur})")
            return 1
        if fmt_hms(3661) != "01:01:01":
            progress("SELF-TEST FAIL: time formatting")
            return 1

        # Write sample outputs with placeholder structure (no Whisper download required for structure test)
        segs = [Segment(0.0, 1.0, "(self-test placeholder - not real speech)")]
        labels = ["Unverified"]
        choice = ModelChoice("base", "cpu", "int8", "SELF-TEST", [])
        tpath = OUTPUT_DIR / TRANSCRIPT_NAME
        npath = OUTPUT_DIR / NOTES_NAME
        write_transcript(tpath, "synthetic_tone.wav", dur, "mixed / not verified", segs, labels)
        write_meeting_notes(
            npath,
            [],
            None,
            False,
            ["Self-test used synthetic tone; not a speech-quality result."],
            choice,
            False,
        )

        # Validate existence, UTF-8, headings - do not treat as real meeting
        for p in (tpath, npath):
            if not p.is_file() or p.stat().st_size < 20:
                progress(f"SELF-TEST FAIL: missing/small file {p.name}")
                return 1
            text = p.read_text(encoding="utf-8")
            if p.name == TRANSCRIPT_NAME:
                if "MEETING TRANSCRIPT" not in text or "Speaker labels are estimates" not in text:
                    progress("SELF-TEST FAIL: Transcript.txt format")
                    return 1
            if p.name == NOTES_NAME:
                for h in (
                    "# MEETING OVERVIEW",
                    "# ENGLISH TRANSCRIPT (DRAFT)",
                    "# DECISIONS (TEACHER REVIEW)",
                    "# ACTION ITEMS (TEACHER REVIEW)",
                    "# WARNINGS",
                ):
                    if h not in text:
                        progress(f"SELF-TEST FAIL: missing heading {h}")
                        return 1

        only = sorted(x.name for x in OUTPUT_DIR.iterdir() if x.is_file())
        if only != [NOTES_NAME, TRANSCRIPT_NAME] and set(only) != {NOTES_NAME, TRANSCRIPT_NAME}:
            progress(f"SELF-TEST WARN: output/ contains {only}")

    progress("SELF-TEST PASS: decode, time format, and output file structure OK.")
    progress("This does NOT prove transcription accuracy (tone is not speech).")
    progress(f"Placeholder files are in {OUTPUT_DIR} - replace them by running on a real recording.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local meeting transcription (Cantonese / Mandarin / English)."
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="SPEED mode: one model tier lower (faster, less accurate).",
    )
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODELS,
        default=None,
        help="Advanced: force Whisper model name (teachers usually skip this).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic decode/output self-test (not speech quality).",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Advanced: process only the first N minutes.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Advanced: skip interactive confirms (use carefully).",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list_input_recordings()
    if not files:
        progress(
            "No recording found.\n"
            f"Please put ONE audio/video file into this folder:\n  {INPUT_DIR}\n"
            "Supported types: .mp3 .wav .m4a .mp4 .mov\n"
            "Then run again (double-click run.bat, or: sh run.sh)."
        )
        return 0
    if len(files) > 1:
        progress("Please keep only ONE recording in the input folder. Found:")
        for f in files:
            progress(f"  - {f.name}")
        progress("Remove or move the extras, then run again. (Nothing was processed.)")
        return 0

    path = files[0]
    if not args.yes:
        ans = input(
            f"Found recording: {path.name}\n"
            "Process this file now? The original will not be renamed or deleted. [y/N]: "
        ).strip().lower()
        if ans not in ("y", "yes"):
            progress("Cancelled. No output written.")
            return 0

    mode = "SPEED" if args.speed else "QUALITY"
    max_seconds = args.minutes * 60.0 if args.minutes else None
    return process_file(path, mode, args.model, max_seconds, skip_confirm=args.yes)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        raise SystemExit(130)
