# Fish Classification – Application Running & Deployment Guide

## Overview

- Local web app: run enhanced Flask UI with detection, classification, segmentation, face marking.
- Cloud API: deploy to Modal as a web function exposing a JSON API.
- Models and static assets are included in the project layout.

## Local Development (Flask UI)

- Create venv and install deps:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r scripts/requirements.txt`
- Run the app:
  - `python scripts/web_app.py`
- Open browser:
  - `http://localhost:5001`

## API (Modal Cloud)

- Install and auth:
  - `pip install modal`
  - `python3 -m modal setup`
- Start dev server (watch mode, ephemeral URL):
  - `modal serve scripts/deploy_modal.py`
- Production deploy (stable URL):
  - `modal deploy scripts/deploy_modal.py`
- API endpoint:
  - `POST /api` on your `*.modal.run` base URL
  - JSON body:
    - `image_b64` (base64 of image)
    - `filename` (optional)
    - `pixels_per_cm` (optional float)
    - `length_type` (`AUTO` | `TL` | `FL` | `LJFL`)
    - `girth_factor` (float, default 3.1416)
- Response JSON fields (same as local web_app):
  - `success` (bool), `fish_count` (int)
  - `fish[]`: `fish_id`, `species`, `accuracy`, `confidence`, `box`
  - `segmentation.points`, `face_box`
  - `measurements`: `length_type`, `length_px/cm`, `girth_px/cm`, optional rule ranges
  - `crop_mask_image`: relative path under `static/`
  - `annotated_image`: relative path under `static/`
- Static files:
  - `GET /static/<relative_path>` on the same base URL

## Performance Tips

- Use environment variables to control workload:
  - `FAST_MODE=1` disables segmentation and face detection
  - `DISABLE_SEGMENTATION=1`, `DISABLE_FACE=1`, `DISABLE_CLASSIFICATION=1` to selectively disable
- Deploy with GPU in Modal function for faster Torch/YOLO:
  - `gpu="T4"` is configured in `scripts/deploy_modal.py`
- Keep-warm containers:
  - `keep_warm=1` pre-warms one container to reduce cold-start
- Reduce rebuilds:
  - Use `modal deploy` for production; avoid frequent file changes while serving

## Example: CLI Test

- Base64 encode and post:
  - `IMG_B64=$(base64 -i /path/to/fish.jpg | tr -d '\n')`
  - `curl -X POST 'https://<your-app>.modal.run/api' -H 'Content-Type: application/json' -d '{"image_b64":"'"$IMG_B64"'","filename":"fish.jpg","pixels_per_cm":null,"length_type":"AUTO","girth_factor":3.1416}'`

## Project Paths

- `scripts/web_app.py` – local UI, processing pipeline
- `scripts/deploy_modal.py` – Modal web function (ASGI/Starlette)
- `static/uploads|results|tmp` – working directories
- `models/*` – detection, segmentation, classification, face detector

## Test with Example Image

- Use `fish.jpeg` in the project root for quick tests.
- Base64 encode it:
  - `IMG_B64=$(base64 -i fish.jpeg | tr -d '\n')`
- Post to the API:
  - `curl -X POST 'https://<your-app>.modal.run/api' -H 'Content-Type: application/json' -d '{"image_b64":"'"$IMG_B64"'","filename":"fish.jpg","pixels_per_cm":null,"length_type":"AUTO","girth_factor":3.1416}'`

## Notes

- The API response matches the local UI pipeline structure for easy integration.
- Static image paths from the response can be fetched via `/static/...` on the same domain.
