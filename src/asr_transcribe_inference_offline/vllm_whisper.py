
from collections import defaultdict
from dataclasses import asdict
import os
import fastapi
import logging

import torch
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset

os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

model_name = "./models/distill-whisper-th-small"

audio_count = 1

def create_transcribe_prompt():
    return [
        "<|startoftranscript|>",
        "<|th|>",
        "<|transcribe|>",
        "<|notimestamps|>"
    ]
    

def transcribe():
    prompt = " ".join(create_transcribe_prompt())

    default_limits = {"image": 0, "video": 0, "audio": 0}
    
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=448,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )
    
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    
    
    engine_args = asdict(engine_args)
    llm = LLM(**engine_args)
    
    print(llm)
    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=64
    )
    
    audio_assets = [AudioAsset("audio")]
   
    mm_data = defaultdict()
    if audio_count > 0:
        mm_data = {
            "audio": [
                asset.audio_and_sample_rate for asset in audio_assets[:audio_count]
            ]
        }
        
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
    )
    
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

