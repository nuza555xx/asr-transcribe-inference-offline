# import time
# from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import logging
import time
from fastapi import FastAPI, File, UploadFile
import numpy as np
import scipy
from faster_whisper.vad import get_speech_timestamps, VadOptions
from asr_transcribe_inference_offline.faster_whisper_large import transcribe_whisper_large_segment
from asr_transcribe_inference_offline.vllm_whisper import transcribe
from asr_transcribe_inference_offline.whisper_th_small import transcribe_whisper_small_segment
import soundfile as sf
from typing import Any, List

MAX_WORKERS = 2 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

def chunk_audio(waveform: np.ndarray[Any, np.dtype[np.float64]], sample_rate: int = 16000):
    if len(waveform.shape) == 2:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != 16000:
        num_samples = round(len(waveform) * float(16000) / sample_rate)
        waveform = scipy.signal.resample(waveform, num_samples)
        sample_rate = 16000  # Update sample_rate after resampling

    vad_options = VadOptions(
        threshold=0.4,
        min_speech_duration_ms=400,
        max_speech_duration_s=10,
        min_silence_duration_ms=500,
        speech_pad_ms=300
    )

    # Ensure waveform is float32 for ONNXRuntime compatibility
    waveform = waveform.astype(np.float32)

    speech_timestamps = get_speech_timestamps(
        waveform, sampling_rate=sample_rate, return_timestamps=True, vad_options=vad_options
    )

    return speech_timestamps


@app.post("/transcribe/distill-whisper-th-small")
async def transcribe_audio(file: UploadFile = File(...), lang: str = "th"):
    total_start = time.time()
    logger.info(f"Received file: {file.filename} | Target language: {lang}")

    audio_bytes = await file.read()
    audio_file = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_file)

    speech_timestamps = chunk_audio(waveform, sample_rate)
    logger.info(f"[Chunking] found: {len(speech_timestamps)}")

    transcriptions_list = [None] * len(speech_timestamps)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(transcribe_whisper_small_segment, idx, waveform[int(seg['start']): int(seg['end'])])
            for idx, seg in enumerate(speech_timestamps)
        ]
        for future in as_completed(futures):
            idx, transcription = future.result()
            transcriptions_list[idx] = transcription

    transcriptions = " ".join(transcriptions_list)
    logger.info(f"[Total] End-to-end time: {time.time() - total_start:.2f}s")

    return {"transcription": transcriptions}

@app.post("/transcribe/faster-whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    total_start = time.time()
    logger.info(f"Received file: {file.filename}")

    audio_bytes = await file.read()
    audio_file = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_file)

    speech_timestamps = chunk_audio(waveform, sample_rate)
    logger.info(f"[Chunking] found: {len(speech_timestamps)}")

    transcription = ""
    for idx, seg in enumerate(speech_timestamps):
        logger.info(f"[Chunking] Processing : {idx}")
        audio = waveform[int(seg['start']): int(seg['end'])]
        transcription += transcribe_whisper_large_segment(audio)

    logger.info(f"[Total] End-to-end time: {time.time() - total_start:.2f}s")

    return {"transcription": transcription}

@app.post("/transcribe/vllm-whisper")
def transcribe_audio(file: UploadFile = File(...)):
    total_start = time.time()
    
    transcription = transcribe()

    logger.info(f"[Total] End-to-end time: {time.time() - total_start:.2f}s")

    return {"transcription": transcription}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.asr_transcribe_inference_offline.main", host="0.0.0.0", port=8000)