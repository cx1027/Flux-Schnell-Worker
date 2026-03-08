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

        try:
            # Call pipeline
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Handle different return types from FluxPipeline
            # Some versions return an object with .images, others return a list/tuple directly
            images = None
            if hasattr(result, 'images'):
                # Standard case: result is an object with images attribute
                images = result.images
            elif isinstance(result, (list, tuple)):
                # Pipeline returned a list/tuple directly
                # If it's a tuple, it might be (images, ...) or just images
                if isinstance(result, tuple) and len(result) > 0:
                    # Check if first element is a list of images
                    if isinstance(result[0], (list, tuple)) and len(result[0]) > 0:
                        if isinstance(result[0][0], Image.Image):
                            images = result[0]
                        else:
                            images = result
                    elif isinstance(result[0], Image.Image):
                        # Tuple of Image objects
                        images = result
                    else:
                        # Try to use the whole result
                        images = result
                else:
                    images = result
            else:
                raise ValueError(
                    f"Pipeline returned unexpected result type: {type(result)}, "
                    f"hasattr(images): {hasattr(result, 'images') if hasattr(result, '__dict__') else 'N/A'}"
                )
            
            # Validate that we have at least one image
            if images is None:
                raise ValueError("Failed to extract images from pipeline result")
            
            if not images or len(images) == 0:
                raise ValueError("Pipeline returned no images")
            
            image: Image.Image = images[0]
            
            # Validate that we got a PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            # Ensure we return a tuple of exactly 2 values
            return (img_bytes, int(seed))
        except Exception as e:
            # Re-raise the exception so it can be caught by the handler
            # Include the original exception type and message for debugging
            error_msg = str(e)
            error_type = type(e).__name__
            raise RuntimeError(
                f"Image generation failed: [{error_type}] {error_msg}"
            ) from e
        finally:
            # Clear CUDA cache after generation to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    # Use sequential CPU offload for more aggressive memory management
    # This offloads models to CPU one at a time, using less GPU memory
    pipe.enable_sequential_cpu_offload()

    # Enable memory-efficient attention
    pipe.enable_xformers_memory_efficient_attention()

    # Enable VAE slicing to process VAE decoder in slices (reduces memory)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # Enable VAE tiling for large images (processes in tiles)
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    # Disable safety checker if present to simplify output.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    return FluxGenerator(pipe=pipe)

