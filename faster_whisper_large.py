import logging
import numpy as np
import torch
from faster_whisper import WhisperModel

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load Faster-Whisper model
model = WhisperModel(
    "./biodatlab-whisper-th-large-v3-faster",
    device="cuda",
    compute_type="float16",
    device_index=0,
    cpu_threads=2,
    num_workers=8,
)
logger.info(f"Loaded Faster-Whisper model on device: {torch.cuda.get_device_capability()}, compute_type: float16")

logger.info(f"Cuda available: {torch.cuda.is_available()}")

# Transcribe segment
def transcribe_whisper_large_segment(chunk: np.ndarray) -> str:
    segments, _ = model.transcribe(
        chunk,
        language="th",
        beam_size=5
    )
    
    transcription = " ".join([seg.text.strip() for seg in segments])
    return transcription,
