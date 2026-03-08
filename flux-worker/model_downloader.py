import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from diffusers import FluxPipeline
from PIL import Image


MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
TORCH_DTYPE = torch.float16
DEVICE = os.environ.get("DEVICE", "cuda")
VOLUME_CHECKPOINTS_DIR = os.environ.get("VOLUME_CHECKPOINTS_DIR", "/runpod-volume/checkpoints")


@dataclass
class FluxGenerator:
    pipe: FluxPipeline

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        """
        Generate a single image and return it as PNG bytes plus the used seed.
        This keeps the core generation logic independent from any transport
        details (base64, object storage, etc.).
        """
        if seed is None:
            seed = torch.seed() % 2**31

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        image: Image.Image = images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        return img_bytes, int(seed)


def ensure_main_model(model_id: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
    """
    Download the model to the specified cache directory if it doesn't exist.
    This function is called at container startup to ensure the model is available.
    
    HuggingFace models are cached in a specific structure:
    - If HF_HOME is set, models go to: HF_HOME/hub/models--{org}--{model_name}/
    - We set HF_HOME to the cache_dir so models persist in the RunPod volume.
    
    Args:
        model_id: HuggingFace model ID (defaults to MODEL_ID env var)
        cache_dir: Directory to cache the model (defaults to VOLUME_CHECKPOINTS_DIR)
    """
    if model_id is None:
        model_id = MODEL_ID
    
    if cache_dir is None:
        cache_dir = VOLUME_CHECKPOINTS_DIR
    
    # Check if cache directory exists
    if cache_dir and not os.path.isdir(cache_dir):
        print(f"WARNING: Cache directory does not exist: {cache_dir}")
        print("Falling back to default HuggingFace cache")
        cache_dir = None
    
    # Check if model already exists by trying to load it
    if cache_dir:
        # Set HF_HOME to use our custom cache directory
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = cache_dir
        
        # Check if model exists in cache
        # HuggingFace cache structure: {HF_HOME}/hub/models--{org}--{model_name}/
        org, name = model_id.split("/", 1)
        cache_model_dir = os.path.join(cache_dir, "hub", f"models--{org}--{name.replace('/', '--')}")
        
        if os.path.isdir(cache_model_dir):
            print(f"Model already exists in cache: {cache_model_dir}")
            # Restore original HF_HOME if it was set
            if original_hf_home:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            return
        
        print(f"Downloading model '{model_id}' to cache directory: {cache_dir}")
        try:
            # Download the model (this will use HF_HOME we just set)
            FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=TORCH_DTYPE,
            )
            print(f"Model downloaded successfully to: {cache_model_dir}")
        finally:
            # Restore original HF_HOME if it was set
            if original_hf_home:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
    else:
        # Use default HuggingFace cache
        print(f"Downloading model '{model_id}' to default HuggingFace cache...")
        FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
        )
        print("Model downloaded successfully to default HuggingFace cache")


def load_model() -> FluxGenerator:
    """
    Load Flux Schnell model in fp16 and return a simple wrapper object.
    This function is executed once at container startup.
    It will load from the volume if available, otherwise from default cache.
    """
    # Set HF_HOME to volume directory if it exists, so HuggingFace will use it
    if VOLUME_CHECKPOINTS_DIR and os.path.isdir(VOLUME_CHECKPOINTS_DIR):
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = VOLUME_CHECKPOINTS_DIR
        print(f"Using volume cache directory: {VOLUME_CHECKPOINTS_DIR}")
    else:
        print("Volume directory not found, using default HuggingFace cache")
    
    try:
        # Load the model (it will use HF_HOME if set, otherwise default cache)
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
        )
    finally:
        # Restore original HF_HOME if we changed it
        if VOLUME_CHECKPOINTS_DIR and os.path.isdir(VOLUME_CHECKPOINTS_DIR):
            if original_hf_home:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]

    # pipe = pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()

    pipe.enable_xformers_memory_efficient_attention()

    # Disable safety checker if present to simplify output.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    return FluxGenerator(pipe=pipe)

