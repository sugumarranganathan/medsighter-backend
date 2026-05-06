from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
import io
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

MEDICINE_KEYWORDS = [
    'gudcef',
    'dolo',
    'crocin',
    'augmentin',
    'calpol',
    'paracetamol',
    'cefpodoxime',
    'amoxicillin',
    'cetirizine'
]


def preprocess_image(image):
    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return processed


def extract_expiry(text):
    patterns = [
        r'EXP[:\s-]*(\d{2}[/-]\d{4})',
        r'EXPIRY[:\s-]*(\d{2}[/-]\d{4})',
        r'USE BEFORE[:\s-]*(\d{2}[/-]\d{4})',
        r'(\d{2}[/-]\d{4})'
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

    return 'Not Found'


def extract_medicine(text):
    lines = [
        line.strip()
        for line in text.split('\n')
        if line.strip()
    ]

    best_match = None

    for line in lines:
        lower = line.lower()

        for keyword in MEDICINE_KEYWORDS:
            if keyword in lower:
                best_match = line.upper()
                break

    if best_match:
        return best_match

    return (
        lines[0].upper()
        if lines
        else 'MEDICINE NOT DETECTED'
    )


@app.get('/')
def home():
    return {
        'message': 'PaddleOCR Backend Running'
    }


@app.post('/ocr')
async def run_ocr(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert('RGB')

    processed = preprocess_image(
        image
    )

    result = ocr.ocr(processed)

    detected_text = []

    for block in result:
        if block:
            for line in block:
                detected_text.append(
                    line[1][0]
                )

    full_text = '\n'.join(
        detected_text
    )

    medicine_name = extract_medicine(
        full_text
    )

    expiry_date = extract_expiry(
        full_text
    )

    return {
        'medicineName': medicine_name,
        'expiryDate': expiry_date,
        'rawText': full_text
    }
