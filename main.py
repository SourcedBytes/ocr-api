import json
import os
import re
from typing import Optional, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from google.oauth2 import service_account
from google.cloud import vision
from dotenv import load_dotenv

load_dotenv()

VIN_TRANSLATION = {
    **{str(i): i for i in range(10)},
    **dict(
        zip(
            "ABCDEFGHJKLMNPRSTUVWXYZ",
            [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    ),
}
VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# fullmatch for a VIN-shaped 17-char string
VIN_SHAPE_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

# Common OCR swaps (we'll try variants if checksum fails)
OCR_FIXES = {
    "I": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "B": "8",
    "Z": "2",
}


def is_valid_vin(vin: str) -> bool:
    if len(vin) != 17:
        return False
    vin = vin.upper()
    if any(c in vin for c in "IOQ"):
        return False

    total = 0
    for i, c in enumerate(vin):
        if c not in VIN_TRANSLATION:
            return False
        total += VIN_TRANSLATION[c] * VIN_WEIGHTS[i]

    check = total % 11
    check_char = "X" if check == 10 else str(check)
    return vin[8] == check_char


def normalize(text: str) -> str:
    # keep only alphanumerics; VIN is uppercase
    t = (text or "").upper()
    return re.sub(r"[^A-Z0-9]", "", t)


def generate_variants(vin: str) -> List[str]:
    # original + single-char swaps (cheap and usually enough)
    variants = {vin}
    for i, c in enumerate(vin):
        if c in OCR_FIXES:
            variants.add(vin[:i] + OCR_FIXES[c] + vin[i + 1 :])
    return list(variants)


def extract_valid_vin(raw_text: str) -> Optional[str]:
    """
    Robust VIN extraction:
      - Normalize OCR text to A-Z0-9
      - Slide a 17-char window (supports overlapping candidates)
      - Validate via ISO checksum
      - Try minor OCR fixes if checksum fails
    """
    t = normalize(raw_text)
    if len(t) < 17:
        return None

    # slide across all overlapping 17-char windows
    for i in range(0, len(t) - 16):
        cand = t[i : i + 17]

        # must look like a VIN (no I/O/Q, valid alphabet)
        if not VIN_SHAPE_RE.match(cand):
            continue

        # prefer strict checksum match
        if is_valid_vin(cand):
            return cand

        # try common OCR swaps
        for v in generate_variants(cand):
            if VIN_SHAPE_RE.match(v) and is_valid_vin(v):
                return v

    return None


def make_vision_client():
    api_endpoint = os.getenv("GOOGLE_VISION_ENDPOINT", "eu-vision.googleapis.com")
    creds_path = os.getenv("GOOGLE_VISION_CREDS_JSON")
    jsondata = os.getenv("GOOGLE_VISION_CREDS_JSON_CONTENT")

    if jsondata:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(jsondata.strip().strip("'\""))
        )
        return vision.ImageAnnotatorClient(
            credentials=credentials,
            client_options={"api_endpoint": api_endpoint},
        )

    elif creds_path:
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        return vision.ImageAnnotatorClient(
            credentials=credentials,
            client_options={"api_endpoint": api_endpoint},
        )

    return vision.ImageAnnotatorClient(client_options={"api_endpoint": api_endpoint})



def ocr_google_bytes(image_bytes: bytes) -> str:
    client = make_vision_client()
    image = vision.Image(content=image_bytes)
    resp = client.text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)

    return resp.text_annotations[0].description if resp.text_annotations else ""


app = FastAPI(title="VIN OCR API", version="1.0.0")


class VinOcrResponse(BaseModel):
    vin: Optional[str]
    raw_text: str
    valid: bool


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/vin/ocr", response_model=VinOcrResponse)
async def vin_ocr(file: UploadFile = File(...)):
    if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        raw_text = ocr_google_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

    vin = extract_valid_vin(raw_text)
    return VinOcrResponse(vin=vin, raw_text=raw_text, valid=vin is not None)
