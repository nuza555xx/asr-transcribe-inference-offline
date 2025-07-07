
import logging
import time
from typing import cast
import scipy
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
import torch
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


model_path = "./models/distill-whisper-th-small"
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)
tokenizer = cast(WhisperProcessor, processor.tokenizer)
feature_extractor = cast(WhisperFeatureExtractor, processor.feature_extractor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model loaded and moved to device: {device}")

def create_transcribe_prompt():
    return [
        "<|startoftranscript|>",
        "<|th|>",
        "<|transcribe|>",
        "<|notimestamps|>"
    ]
    
def transcribe_whisper_small_segment(idx: int, chunk: np.ndarray, sample_rate: int = 16000, lang: str = "th"):
        logger.info(f"[Chunking] Processing : {idx}")

        if len(chunk.shape) == 2:
            chunk = np.mean(chunk, axis=1)

        if sample_rate != 16000:
            num_samples = round(len(chunk) * float(16000) / sample_rate)
            chunk = scipy.signal.resample(chunk, num_samples)

        start = time.time()
        input_features, decoder_input_ids = preprocess_audio(chunk)
        logger.info(f"[{idx}] Preprocessing done in {time.time() - start:.2f}s")

        start = time.time()
        generated_ids = run_inference(input_features, decoder_input_ids, lang)
        logger.info(f"[{idx}] Inference done in {time.time() - start:.2f}s")

        start = time.time()
        transcription = postprocess_output(generated_ids)
        logger.info(f"[{idx}] Postprocessing done in {time.time() - start:.2f}s")

        return idx, transcription.strip()

def preprocess_audio(chunk: np.ndarray):
    input_data = feature_extractor(
        chunk,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    prompts = tokenizer.convert_tokens_to_ids(create_transcribe_prompt())
    decoder_input_ids = torch.tensor([prompts], device=device)
    
    
    input_features = input_data.input_features.to(device)

    return input_features, decoder_input_ids

def run_inference(input_features, decoder_input_ids, lang: str):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    generated_ids = model.generate(
         input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        use_cache=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        forced_decoder_ids=forced_decoder_ids,
    )
    return generated_ids

def postprocess_output(generated_ids):
    if generated_ids.shape[0] > 1:
        transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
    else:
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)