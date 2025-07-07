# ... (keep all previous stages as in your original Dockerfile)

#################### FASTAPI SERVER ####################
FROM vllm-base AS vllm-fastapi

# Install FastAPI and Uvicorn
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi uvicorn

# Copy your FastAPI app code into the image
# (Assuming your FastAPI app is in app/ directory)
COPY app /vllm-workspace/app

# Set working directory
WORKDIR /vllm-workspace

# Set environment variable if needed
ENV VLLM_USAGE_SOURCE production-docker-image

# Run FastAPI app with uvicorn
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#################### FASTAPI SERVER ####################
