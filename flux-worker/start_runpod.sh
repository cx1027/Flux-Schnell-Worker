#!/bin/bash
set -e

# RunPod volume mount point
VOLUME_CHECKPOINTS_DIR="/runpod-volume/checkpoints"
MODEL_ID="${FLUX_MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

echo "=========================================="
echo "RunPod Worker Startup Script"
echo "=========================================="
echo "Model ID: ${MODEL_ID}"
echo "Volume checkpoints directory: ${VOLUME_CHECKPOINTS_DIR}"
echo ""

# Check if volume directory exists
if [ ! -d "/runpod-volume" ]; then
    echo "WARNING: /runpod-volume directory not found!"
    echo "Make sure you have mounted a RunPod Network Volume to /runpod-volume"
    echo "Falling back to default HuggingFace cache location..."
    VOLUME_CHECKPOINTS_DIR=""
fi

# Create checkpoints directory if volume exists
if [ -d "/runpod-volume" ] && [ ! -d "${VOLUME_CHECKPOINTS_DIR}" ]; then
    echo "Creating checkpoints directory: ${VOLUME_CHECKPOINTS_DIR}"
    mkdir -p "${VOLUME_CHECKPOINTS_DIR}"
fi

# Check if model exists in volume (HuggingFace cache structure: hub/models--{org}--{model_name}/)
MODEL_EXISTS=false
if [ -d "${VOLUME_CHECKPOINTS_DIR}" ]; then
    # Convert model ID to HuggingFace cache directory name
    # e.g., "black-forest-labs/FLUX.1-schnell" -> "models--black-forest-labs--FLUX.1-schnell"
    ORG_NAME=$(echo "${MODEL_ID}" | cut -d'/' -f1)
    MODEL_NAME=$(echo "${MODEL_ID}" | cut -d'/' -f2- | tr '/' '--')
    CACHE_MODEL_DIR="${VOLUME_CHECKPOINTS_DIR}/hub/models--${ORG_NAME}--${MODEL_NAME}"
    
    if [ -d "${CACHE_MODEL_DIR}" ]; then
        echo "Model found in volume: ${CACHE_MODEL_DIR}"
        MODEL_EXISTS=true
    else
        echo "Model not found in volume, will download..."
        echo "Expected cache directory: ${CACHE_MODEL_DIR}"
    fi
else
    echo "Volume directory not available, will use default HuggingFace cache..."
fi

# Download model if needed
if [ "$MODEL_EXISTS" = false ]; then
    echo ""
    echo "Downloading model to volume..."
    export VOLUME_CHECKPOINTS_DIR="${VOLUME_CHECKPOINTS_DIR}"
    python3 -c "
from model_downloader import ensure_main_model
import os
volume_dir = os.environ.get('VOLUME_CHECKPOINTS_DIR', '')
model_id = os.environ.get('FLUX_MODEL_ID', 'black-forest-labs/FLUX.1-schnell')
ensure_main_model(model_id=model_id, cache_dir=volume_dir if volume_dir else None)
print('Model download completed!')
"
fi

# Export VOLUME_CHECKPOINTS_DIR for handler.py to use
export VOLUME_CHECKPOINTS_DIR="${VOLUME_CHECKPOINTS_DIR}"

echo ""
echo "Starting RunPod handler..."
echo "=========================================="

# Start the handler
exec python3 handler.py
