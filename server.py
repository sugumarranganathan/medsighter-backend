from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
import io
import re
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR Engine
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en'
)

# Medicine keywords
MEDICINE_KEYWORDS = [
    'gudcef',
    'dolo',
    'crocin',
    'augmentin',
    'calpol',
    'paracetamol',
    'cefpodoxime',
    'amoxicillin',
    'cetirizine',
    'azithromycin',
    'pantoprazole',
    'metformin',
    'atorvastatin'
]

# Ignore OCR junk
IGNORE_WORDS = [
    'schedule',
    'warning',
    'marketed',
    'manufactured',
    'composition',
    'contains',
    'dosage',
    'tablet',
    'capsule',
    'alkem',
    'prescription',
    'caution',
    'storage',
    'batch',
    'mfg',
    'mrp'
]


def preprocess_image(image):
    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Upscale image
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Noise reduction
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # Sharpen
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    # Threshold
    processed = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return processed


def clean_text(text):
    return re.sub(
        r'[^A-Za-z0-9\s/-]',
        '',
        text
    ).strip()


def extract_expiry(text):
    patterns = [
        r'EXP[:\s-]*(\d{2}[/-]\d{4})',
        r'EXPIRY[:\s-]*(\d{2}[/-]\d{4})',
        r'USE BEFORE[:\s-]*(\d{2}[/-]\d{4})',
        r'BEST BEFORE[:\s-]*(\d{2}[/-]\d{4})',
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


def score_line(line):
    score = 0

    lower = line.lower()

    # Keyword bonus
    for keyword in MEDICINE_KEYWORDS:
        if keyword in lower:
            score += 100

    # Medicine style text
    if re.search(r'[A-Za-z]+\s?\d+', line):
        score += 40

    if len(line) > 5:
        score += 10

    # Ignore junk
    for word in IGNORE_WORDS:
        if word in lower:
            score -= 80

    return score


def extract_medicine(text):
    lines = [
        clean_text(line)
        for line in text.split('\n')
        if clean_text(line)
    ]

    best_line = None
    best_score = -999

    for line in lines:
        current_score = score_line(line)

        if current_score > best_score:
            best_score = current_score
            best_line = line

    if best_line:
        best_line = re.sub(
            r'\s+',
            ' ',
            best_line
        )

        return best_line.upper()

    return 'MEDICINE NOT DETECTED'


def check_expired(expiry):
    if expiry == 'Not Found':
        return False

    try:
        clean = expiry.replace('-', '/')

        month, year = clean.split('/')

        month = int(month)
        year = int(year)

        if year < 100:
            year += 2000

        expiry_date = datetime(
            year,
            month,
            1
        )

        return datetime.now() > expiry_date

    except:
        return False


@app.get('/')
def home():
    return {
        'message': 'Advanced Pharmaceutical OCR Running'
    }


@app.post('/ocr')
async def run_ocr(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert('RGB')

    processed = preprocess_image(image)

    result = ocr.ocr(processed)

    detected_text = []

    for block in result:
        if block:
            for line in block:
                text = line[1][0]

                if len(text.strip()) > 1:
                    detected_text.append(text)

    full_text = '\n'.join(detected_text)

    medicine_name = extract_medicine(full_text)

    expiry_date = extract_expiry(full_text)

    is_expired = check_expired(expiry_date)

    return {
        'medicineName': medicine_name,
        'expiryDate': expiry_date,
        'isExpired': is_expired,
        'rawText': full_text
    }
