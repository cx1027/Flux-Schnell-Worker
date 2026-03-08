import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from diffusers import FluxPipeline


MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
TORCH_DTYPE = torch.float16
VOLUME_CHECKPOINTS_DIR = os.environ.get("VOLUME_CHECKPOINTS_DIR", "/runpod-volume/checkpoints")


@dataclass
class FluxGenerator:
    pipe: FluxPipeline

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        if seed is None:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        try:
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=256,
            )
            image = result.images[0]
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {type(e).__name__}: {e}") from e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), int(seed)


def ensure_main_model(model_id: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
    """
    Ensure the main Flux model is present in the cache directory.

    This is mainly used at container start to pre-download the model into
    the RunPod volume so subsequent cold starts are faster.
    """
    if model_id is None:
        model_id = MODEL_ID

    if cache_dir is None:
        cache_dir = VOLUME_CHECKPOINTS_DIR

    # If no cache_dir or it doesn't exist, just trigger a normal load
    if not cache_dir or not os.path.isdir(cache_dir):
        print(f"Cache directory not found, downloading '{model_id}' to default HuggingFace cache...")
        _ = FluxPipeline.from_pretrained(model_id, torch_dtype=TORCH_DTYPE)
        print("Model download complete (default cache).")
        return

    # Use the same HF_HOME handling as load_model so cache is under volume
    original_hf_home = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = cache_dir

    try:
        # HuggingFace cache structure: {HF_HOME}/hub/models--{org}--{name}/
        try:
            org, name = model_id.split("/", 1)
        except ValueError:
            org, name = "models", model_id.replace("/", "--")

        cache_model_dir = os.path.join(
            cache_dir,
            "hub",
            f"models--{org}--{name.replace('/', '--')}",
        )

        print(f"Expected cache directory: {cache_model_dir}")

        if os.path.isdir(cache_model_dir):
            print(f"Model already exists in cache: {cache_model_dir}")
            return

        print("Downloading model to volume...")
        _ = FluxPipeline.from_pretrained(model_id, torch_dtype=TORCH_DTYPE)
        print(f"Model downloaded successfully to: {cache_model_dir}")
    finally:
        if original_hf_home is not None:
            os.environ["HF_HOME"] = original_hf_home
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]


def load_model() -> FluxGenerator:
    original_hf_home = os.environ.get("HF_HOME")

    try:
        if VOLUME_CHECKPOINTS_DIR and os.path.isdir(VOLUME_CHECKPOINTS_DIR):
            os.environ["HF_HOME"] = VOLUME_CHECKPOINTS_DIR

        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
        )
    finally:
        if original_hf_home is not None:
            os.environ["HF_HOME"] = original_hf_home
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]

    pipe.enable_sequential_cpu_offload()

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    return FluxGenerator(pipe=pipe)
