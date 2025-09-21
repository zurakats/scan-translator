import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
import easyocr
import numpy as np
from googletrans import Translator
import pathlib
import re

pathlib.PosixPath = pathlib.WindowsPath

model_path = "yolo-model/bubble-detector-new/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github', force_reload=True)
model.conf = 0.75

translator = Translator()

def process_language(sourceLang):
    try:        
        if sourceLang == "jp":
            return "ja"
        elif sourceLang == "ko":
            return "ko"
        elif sourceLang == "ch_sim":
            return "zh_cn"
        elif sourceLang == "en":
            return "en"
            
    except Exception as e:
        print(f"⚠️ Error saat memproses OCR: {e}")
        raise
    
def get_language_codes(sourceLang):
    mapping = {
        "jp": ("jp", "ja"), # Jepang
        "ko": ("ko", "ko"), # Korea
        "ch_sim": ("ch_sim", "zh-cn"), # Mandarin Sederhana
        "ch_tra": ("ch_tra", "zh-tw"), # Mandarin Tradisional
        "en": ("en", "en") # Inggris
    }

    if sourceLang not in mapping:
        raise ValueError(f"Unsupported language code: {sourceLang}")

    return mapping[sourceLang] 

def selectOCR(sourceLang):
    easyocr_code, translate_code = get_language_codes(sourceLang)
    
    if (easyocr_code == "jp"):
        return MangaOcr(), 0, translate_code
    else:
        return easyocr.Reader([f'{easyocr_code}']), 1, translate_code

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

def process_image(image_path, ocr, index, translate_code, targetLang):
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
        
        if (index == 0):
            text = ocr(cropped) or ""
        else:
            cropped_np = np.array(cropped)
            result = ocr.readtext(cropped_np, detail=0)
            text = " ".join(result) or ""

        cleaned_text = clean_ocr_text(text) or ""

        if cleaned_text.strip():
            translated = translator.translate(cleaned_text, src=translate_code, dest=targetLang).text
            translated = re.sub(r'\bLai\b', '!', translated)
        else:
            translated = "[Teks tidak terbaca]"

        print("OCR Result:", text)
        print("Cleaned Text:", cleaned_text)
        print("Translated:", translated)

        font_size = 32
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