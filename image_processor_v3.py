import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from transformers import pipeline as hf_pipeline
import pathlib
import re
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Baca API Key dari .env
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY tidak ditemukan di file .env")

GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'

pathlib.PosixPath = pathlib.WindowsPath

model_path = "yolo-model/bubble-detector-new/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github', force_reload=True)

ocr = MangaOcr()
translator = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
translator_id = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")


def translate_with_gemini(japanese_text):
    prompt = f"Terjemahkan teks Jepang ini ke bahasa Indonesia. Hasil terjemahan ini digunakan untuk menerjemahkan berbagai macam komik. Tidak membutuhkan response tambahan berupa pertanyaan, terjemahkan saja teks Jepang ke bahasa Indonesia, jangan berikan respon berupa romaji:\n\n{japanese_text}"

    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(GEMINI_URL, headers=headers, json=data)
    print(f"[DEBUG] Status Code: {response.status_code}")
    print(f"[DEBUG] Response Text: {response.text}")

    if response.status_code == 200:
        try:
            result = response.json()
            translated_text = result['candidates'][0]['content']['parts'][0]['text']
            return translated_text
        except (KeyError, IndexError):
            return "[Terjemahan gagal diproses]"
    else:
        return "[Terjemahan gagal]"

def clean_ocr_text(ocr_text):
    if not ocr_text:
        return ""
    cleaned_text = re.sub(r'\bLai\b', '!', ocr_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def wrap_text(text, draw, font, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def process_image(image_path):
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    results = model(img_rgb)
    detections = results.pandas().xyxy[0]
    font_path = "C:\\Windows\\Fonts\\arial.ttf"

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        box_width = x2 - x1
        box_height = y2 - y1
        cropped = img_pil.crop((x1, y1, x2, y2))

        text = ocr(cropped) or ""
        cleaned_text = clean_ocr_text(text) or ""

        if cleaned_text.strip():
            translated = translate_with_gemini(cleaned_text)
            translated = re.sub(r'\bLai\b', '!', translated)
        else:
            translated = "[Teks tidak terbaca]"

        print("OCR Result:", text)
        print("Cleaned Text:", cleaned_text)
        print("Translated:", translated)

        font_size = 24
        while font_size >= 10:
            font = ImageFont.truetype(font_path, font_size)
            lines = wrap_text(translated, draw, font, box_width)
            line_spacing_factor = 1.2

            ascent, descent = font.getmetrics()
            line_height = int((ascent + descent) * line_spacing_factor)

            total_height = line_height * len(lines)

            if total_height <= box_height:
                break
            font_size -= 1

        draw.rectangle([x1, y1, x2, y2], fill="white")

        current_y = y1 + (box_height - total_height) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            text_x = x1 + (box_width - line_width) // 2
            draw.text((text_x, current_y), line, font=font, fill="black")
            current_y += line_height

    return img_pil
