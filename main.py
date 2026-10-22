import io
import os
import tempfile
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Song Analyzer", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_BYTES = 40 * 1024 * 1024
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
CHORD_TEMPLATES = {}
for root, name in enumerate(NOTE_NAMES):
    for quality, intervals in [
        ("", [0, 4, 7]), ("m", [0, 3, 7]), ("dim", [0, 3, 6]),
        ("aug", [0, 4, 8]), ("sus2", [0, 2, 7]), ("sus4", [0, 5, 7])
    ]:
        v = np.zeros(12, dtype=float)
        for interval in intervals:
            v[(root + interval) % 12] = 1.0
        CHORD_TEMPLATES[f"{name}{quality}"] = v / np.linalg.norm(v)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def estimate_key(chroma: np.ndarray) -> dict[str, Any]:
    profile = np.mean(chroma, axis=1)
    profile = normalize(profile)
    scores = []
    for root, note in enumerate(NOTE_NAMES):
        for mode, base in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            rotated = np.roll(base, root)
            score = float(np.corrcoef(profile, normalize(rotated))[0, 1])
            if not np.isfinite(score):
                score = 0.0
            scores.append((score, note, mode))
    scores.sort(reverse=True)
    best = scores[0]
    # Difference from the runner-up is a useful rough confidence indicator, not a probability.
    margin = max(0.0, best[0] - scores[1][0])
    confidence = round(min(0.99, max(0.05, 0.5 + margin * 2.5)), 2)
    return {"key": f"{best[1]} {best[2]}", "confidence": confidence}

def estimate_chords(y: np.ndarray, sr: int, hop_length: int, tempo: float) -> list[dict[str, Any]]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    duration = librosa.get_duration(y=y, sr=sr)
    # Use roughly one bar per segment in 4/4; clamp to keep changes responsive.
    seconds_per_bar = 240.0 / max(tempo, 1.0)
    segment_seconds = float(np.clip(seconds_per_bar, 1.0, 4.0))
    step = max(1, int(segment_seconds * sr / hop_length))
    segments = []
    for start_frame in range(0, chroma.shape[1], step):
        end_frame = min(chroma.shape[1], start_frame + step)
        if end_frame <= start_frame:
            continue
        vec = normalize(np.mean(chroma[:, start_frame:end_frame], axis=1))
        if np.linalg.norm(vec) < 1e-8:
            continue
        ranked = sorted(
            ((float(np.dot(vec, template)), name) for name, template in CHORD_TEMPLATES.items()),
            reverse=True
        )
        score, chord = ranked[0]
        start = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end = min(duration, librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length))
        if end <= start:
            continue
        segments.append({
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "chord": chord,
            "confidence": round(float(np.clip(score, 0.0, 1.0)), 2)
        })
    # Merge adjacent identical labels to reduce flicker.
    merged = []
    for item in segments:
        if merged and merged[-1]["chord"] == item["chord"] and item["start"] - merged[-1]["end"] < 0.1:
            merged[-1]["end"] = item["end"]
            merged[-1]["confidence"] = round((merged[-1]["confidence"] + item["confidence"]) / 2, 2)
        else:
            merged.append(item)
    return merged

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = file.filename or "audio.mp3"
    if not filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Upload an MP3, WAV, M4A, OGG, or FLAC audio file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 40 MB.")
    suffix = os.path.splitext(filename)[1] or ".audio"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            temp_path = tmp.name
        # Mono, 22.05 kHz keeps analysis reasonably fast for a hosted MVP.
        y, sr = librosa.load(temp_path, sr=22050, mono=True, duration=600)
        if y.size < sr:
            raise HTTPException(status_code=400, detail="Audio must be at least one second long.")
        hop_length = 512
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=True)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
        if tempo_value <= 0:
            tempo_value = 0.0
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        key_result = estimate_key(chroma)
        chords = estimate_chords(y, sr, hop_length, tempo_value or 120.0)
        duration = float(librosa.get_duration(y=y, sr=sr))
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return {
            "filename": os.path.basename(filename),
            "duration": round(duration, 2),
            "tempo_bpm": round(tempo_value, 1),
            "key": key_result["key"],
            "key_confidence": key_result["confidence"],
            "beats": [round(float(t), 2) for t in beat_times[:3000]],
            "chords": chords,
            "note": "Estimates are generated from the uploaded audio. Full-mix chord recognition can be ambiguous, especially with dense arrangements, alternate tunings, or changing bass notes."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze this audio file. Try a different file. Details: {str(exc)[:180]}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
