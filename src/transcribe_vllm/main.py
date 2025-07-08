import io
from contextlib import asynccontextmanager

import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from vllm import LLM, SamplingParams

# Global variables to hold the model and sampling parameters
llm_engine = None
sampling_params = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager to handle the startup and shutdown events
    of the FastAPI application. This is the recommended way to manage resources
    like machine learning models.
    """
    global llm_engine, sampling_params
    
    print("Loading the transcription model...")
    # Initialize the vLLM engine with the specified model.
    # 'openai/whisper-large-v3' is a powerful model for audio transcription.
    # The task is set to 'generate' as required by vLLM for ASR models.
    llm_engine = LLM(model="biodatlab/distill-whisper-th-small", task="generate", trust_remote_code=True)
    
    # Define the sampling parameters for generation.
    # Temperature 0.0 makes the output deterministic.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    print("Model loaded successfully.")
    
    yield
    
    # Clean up resources on shutdown (optional)
    print("Cleaning up resources...")
    llm_engine = None
    sampling_params = None


# Initialize the FastAPI app with the lifespan event handler
app = FastAPI(
    lifespan=lifespan,
    title="Audio Transcription with vLLM Library",
    description="An API to transcribe audio files using the vLLM library directly.",
    version="2.0.0",
)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file, processes it, and uses the vLLM engine to transcribe it.

    Args:
        file (UploadFile): The audio file to be transcribed.

    Returns:
        JSONResponse: A JSON response containing the transcription text.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again in a moment.")

    try:
        # Read the audio file content as bytes
        audio_bytes = await file.read()

        # Load the audio bytes using librosa.
        # Whisper models expect audio to be at a 16kHz sample rate.
        audio_input, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # Prepare the multi-modal data payload for vLLM
        multi_modal_data = {"audio": [(audio_input, 16000)]}

        # This prompt structure is specific to Whisper models.
        # It instructs the model to perform transcription in English without timestamps.
        # The <|AUDIO|> token is a placeholder that vLLM replaces with the audio data.
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|><|AUDIO|>"

        # Call the vLLM engine to generate the transcription
        outputs = llm_engine.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data
        )

        # Extract the transcribed text from the first output
        transcribed_text = outputs[0].outputs[0].text
        return JSONResponse(content={"transcription": transcribed_text.strip()})

    except Exception as e:
        # Catch potential errors during processing and return an informative message
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {e}")


@app.get("/")
def read_root():
    """
    A simple root endpoint to check if the server is running.
    """
    return {"message": "Welcome to the Audio Transcription API with vLLM!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.transcribe_vllm.main:app", host="0.0.0.0", port=8000)

