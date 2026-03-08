import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
import runpod

from model_downloader import load_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Load model at container startup so it is cached in GPU memory for all requests.
MODEL = load_model()


# ---------------------------------------------------------------------------
# Cloudflare R2 (S3-compatible) configuration
# ---------------------------------------------------------------------------

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None,
)
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_BUCKET_DOMAIN = os.environ.get("R2_PUBLIC_BUCKET_DOMAIN")  # e.g. cdn.example.com
R2_KEY_PREFIX = os.environ.get("R2_KEY_PREFIX", "flux/")


def _create_s3_client() -> boto3.client:
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        raise RuntimeError(
            "Missing one or more required R2 env vars: "
            "R2_ACCOUNT_ID/R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
        )

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


S3_CLIENT = _create_s3_client()


async def upload_png_to_r2(image_bytes: bytes, key: str) -> str:
    """
    Upload PNG bytes to Cloudflare R2 asynchronously using a thread executor.
    Returns the public URL of the uploaded object.
    """
    loop = asyncio.get_running_loop()

    def _put_object() -> None:
        S3_CLIENT.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
        )

    await loop.run_in_executor(None, _put_object)

    if R2_PUBLIC_BUCKET_DOMAIN:
        # Recommended: configure a public domain in front of R2 (via Cloudflare)
        return f"https://{R2_PUBLIC_BUCKET_DOMAIN.rstrip('/')}/{key}"

    # Fallback to direct R2 URL
    base = R2_ENDPOINT_URL.rstrip("/") if R2_ENDPOINT_URL else ""
    return f"{base}/{R2_BUCKET_NAME}/{key}"


def _build_object_key() -> str:
    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    uid = uuid.uuid4().hex
    return f"{R2_KEY_PREFIX.rstrip('/')}/{date_prefix}/{uid}.png"


async def _handle_single(inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = inputs.get("prompt", "")
    if not prompt:
        return {"error": "Missing 'prompt' in input"}

    negative_prompt = inputs.get("negative_prompt")
    width = int(inputs.get("width", 1024))
    height = int(inputs.get("height", 1024))
    num_inference_steps = int(inputs.get("num_inference_steps", 20))
    guidance_scale = float(inputs.get("guidance_scale", 3.5))
    seed = inputs.get("seed")

    loop = asyncio.get_running_loop()

    try:
        # Run heavy model inference in a thread to avoid blocking the event loop.
        image_bytes, used_seed = await loop.run_in_executor(
            None,
            lambda: MODEL.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            ),
        )
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"inference_failed: {e}"}

    object_key = _build_object_key()

    try:
        image_url = await upload_png_to_r2(image_bytes, object_key)
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"upload_failed: {e}"}

    return {
        "seed": used_seed,
        "width": width,
        "height": height,
        "image_url": image_url,
        "object_key": object_key,
        "bucket": R2_BUCKET_NAME,
    }


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless async handler.

    Expected event format:
    {
        "input": {
            "prompt": "a cute cat",
            "negative_prompt": "low quality, blurry",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "seed": 42
        }
    }

    The handler now generates the image, uploads it to Cloudflare R2, and returns
    the public URL instead of a base64 payload. Because the handler is async and
    uses thread executors for blocking work, RunPod can handle concurrent
    requests more efficiently.
    """
    inputs = event.get("input", {}) or {}
    return await _handle_single(inputs)


runpod.serverless.start({"handler": handler})
