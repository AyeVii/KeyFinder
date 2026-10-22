# Song Analyzer — MVP

A deployable web app that accepts audio uploads and estimates tempo, key, beat positions, and a time-aligned chord timeline using real signal analysis.

## Features
- Upload MP3, WAV, M4A, OGG, or FLAC (up to 40 MB)
- BPM estimate and beat markers from librosa beat tracking
- Key estimate from chroma features and key profiles
- Chord timeline from chroma-CQT features matched against major/minor/diminished/augmented/suspended chord templates
- Audio playback and clickable chord segments
- FastAPI backend and single-page frontend

## Run locally
Requires Python 3.11 and FFmpeg.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

## Deploy online with Render
1. Create a GitHub repository and upload the contents of this folder.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml` and builds the Docker image.
4. When deployment completes, open the generated `onrender.com` URL.

For a first public deployment, use a paid instance for dependable CPU and memory. Analysis can take time on long tracks; the MVP limits analysis to the first 10 minutes and each upload to 40 MB.

## Important limitations
These are actual audio-derived estimates, not demo values. Tempo may be reported at half/double time; key confidence is a heuristic rather than a calibrated probability. Chord recognition uses harmonic chroma templates, not a large learned transcription model, so dense mixes and non-triad chords can be misidentified. Validate results against reference tracks before relying on them professionally.

## Production hardening before public launch
- Add user authentication, upload rate limits, and per-user quotas.
- Run analysis in a background job queue for long files.
- Add stronger input validation and malware scanning.
- Store files temporarily or in object storage with explicit deletion/retention policy.
- Add an improved chord model such as a trained audio chord-recognition model, and calibrate confidence.
- Consider an audio separation stage for difficult mixes.
