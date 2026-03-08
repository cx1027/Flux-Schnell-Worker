import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
import runpod

from model_downloader import load_model


MODEL = load_model()

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None,
)
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_BUCKET_DOMAIN = os.environ.get("R2_PUBLIC_BUCKET_DOMAIN")
R2_KEY_PREFIX = os.environ.get("R2_KEY_PREFIX", "flux/")


def _create_s3_client():
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
        return f"https://{R2_PUBLIC_BUCKET_DOMAIN.rstrip('/')}/{key}"

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

    width = int(inputs.get("width", 768))
    height = int(inputs.get("height", 768))
    num_inference_steps = int(inputs.get("num_inference_steps", 4))
    guidance_scale = float(inputs.get("guidance_scale", 0.0))

    seed = inputs.get("seed")

    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: MODEL.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            ),
        )

        if not isinstance(result, tuple) or len(result) != 2:
            return {
                "error": f"inference_failed: generate_image returned unexpected type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
            }

        image_bytes, used_seed = result

    except ValueError as e:
        error_msg = str(e)
        if "not enough values to unpack" in error_msg:
            return {
                "error": f"inference_failed: Unpacking error - {error_msg}. This suggests generate_image() returned an unexpected value."
            }
        return {"error": f"inference_failed: {error_msg}"}
    except Exception as e:
        return {"error": f"inference_failed: {e}"}

    object_key = _build_object_key()

    try:
        image_url = await upload_png_to_r2(image_bytes, object_key)
    except Exception as e:
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
    inputs = event.get("input", {}) or {}
    return await _handle_single(inputs)


runpod.serverless.start({"handler": handler})
