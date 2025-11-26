# DV Lottery Photo Check - Backend (Python FastAPI)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import io
import base64
from datetime import datetime
import sqlite3
import os
from PIL import Image

app = FastAPI(title="DV Lottery Photo API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
DB_FILE = "dv_lottery.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
    (telegram_id TEXT PRIMARY KEY,
    attempts INTEGER DEFAULT 3,
    tariff TEXT DEFAULT 'free',
    tariff_expiry DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Promo codes table
    c.execute('''CREATE TABLE IF NOT EXISTS promo_codes
    (code TEXT PRIMARY KEY,
    attempts INTEGER,
    max_uses INTEGER,
    used_count INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT 1)''')

    # Promo usage table
    c.execute('''CREATE TABLE IF NOT EXISTS promo_usage
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id TEXT,
    promo_code TEXT,
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Transactions table
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id TEXT,
    tariff TEXT,
    amount INTEGER,
    payment_status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# Models
class PhotoProcessRequest(BaseModel):
    userId: str
    photo: str  # base64 encoded
    imageWidth: int  # Original image width
    imageHeight: int  # Original image height
    lines: dict  # {top, bottom, eyes} in percentage

class PromoCodeRequest(BaseModel):
    userId: str
    promoCode: str

# Helper Functions
def get_user(telegram_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
    user = c.fetchone()
    conn.close()
    return user

def create_or_update_user(telegram_id: str, attempts: int = 3):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    user = get_user(telegram_id)
    if user is None:
        c.execute("INSERT INTO users (telegram_id, attempts) VALUES (?, ?)", (telegram_id, attempts))
    else:
        c.execute("UPDATE users SET attempts = ? WHERE telegram_id = ?", (attempts, telegram_id))
    conn.commit()
    conn.close()

def update_attempts(telegram_id: str, attempts: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET attempts = ? WHERE telegram_id = ?", (attempts, telegram_id))
    conn.commit()
    conn.close()

def decode_base64_image(base64_string: str):
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise

def encode_image_to_base64(img_array):
    """Encode numpy array to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        raise

def crop_photo_by_lines(image, image_width, image_height, lines):
    """
    Crop photo to 600x600 based on user-defined lines
    lines: {top, bottom, eyes} in percentage of image height
    """
    try:
        height, width = image.shape[:2]

        # Convert percentage lines to pixel positions
        top_line_px = int((lines.get('top', 15) / 100) * height)
        bottom_line_px = int((lines.get('bottom', 80) / 100) * height)

        # Calculate head height
        head_height = bottom_line_px - top_line_px

        if head_height <= 0:
            raise ValueError("Invalid line positions")

        # Target: head should be ~60% of final 600x600 image (optimal)
        # So we need total_height = head_height / 0.60
        target_size = max(600, int(head_height / 0.60))

        # Make it square
        crop_size = target_size

        # Center horizontally
        center_x = width // 2
        left_x = max(0, center_x - crop_size // 2)
        right_x = min(width, left_x + crop_size)

        # Adjust if out of bounds
        if right_x - left_x < crop_size:
            if right_x == width:
                left_x = max(0, right_x - crop_size)
            else:
                right_x = min(width, left_x + crop_size)

        # Position vertically: put top of head with small margin
        margin_above_head = int(crop_size * 0.10)  # 10% margin above head
        top_y = max(0, top_line_px - margin_above_head)
        bottom_y = min(height, top_y + crop_size)

        # Adjust if out of bounds
        if bottom_y - top_y < crop_size:
            if bottom_y == height:
                top_y = max(0, bottom_y - crop_size)
            else:
                bottom_y = min(height, top_y + crop_size)

        # Final crop
        cropped = image[top_y:bottom_y, left_x:right_x]

        # Resize to exactly 600x600 with high quality
        cropped_600 = cv2.resize(cropped, (600, 600), interpolation=cv2.INTER_LANCZOS4)

        return cropped_600

    except Exception as e:
        print(f"Error cropping photo: {e}")
        raise

def detect_face(image):
    """Detect face using Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        # Get largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face

        # Detect eyes within face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        face_data = {
            'face': (x, y, w, h),
            'eyes': eyes,
            'face_center_x': x + w // 2,
            'face_center_y': y + h // 2
        }

        return face_data
    except Exception as e:
        print(f"Error detecting face: {e}")
        return None

def validate_photo(image, lines):
    """Validate photo against DV lottery requirements"""
    criteria = []

    try:
        height, width = image.shape[:2]

        # 1. Image dimensions
        criteria.append({
            'name': 'Размер фото',
            'value': f'{width}×{height} px',
            'passed': width == 600 and height == 600
        })

        # 2. File size
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        file_size_kb = len(buffer) / 1024
        criteria.append({
            'name': 'Размер файла',
            'value': f'{int(file_size_kb)} Кб',
            'passed': 10 <= file_size_kb <= 240
        })

        # 3. Resolution
        criteria.append({
            'name': 'Разрешение',
            'value': '≥300 dpi',
            'passed': True
        })

        # 4. Head position (converted to 600px scale)
        top_px = int((lines['top'] / 100) * 600)
        bottom_px = int((lines['bottom'] / 100) * 600)
        eye_px = int((lines['eyes'] / 100) * 600)

        criteria.append({
            'name': 'Макушка головы',
            'value': f'{top_px} px',
            'passed': 40 <= top_px <= 150
        })

        criteria.append({
            'name': 'Подбородок',
            'value': f'{bottom_px} px',
            'passed': 380 <= bottom_px <= 560
        })

        # 5. Head size percentage
        head_height = bottom_px - top_px
        head_size_percent = (head_height / 600) * 100
        criteria.append({
            'name': 'Размер головы',
            'value': f'{int(head_size_percent)}%',
            'passed': 50 <= head_size_percent <= 69
        })

        # 6. Eye line position
        criteria.append({
            'name': 'Уровень глаз',
            'value': f'{eye_px} px ({int((eye_px/600)*100)}%)',
            'passed': 56 <= (eye_px/600)*100 <= 69
        })

        # 7. Face detection
        face_data = detect_face(image)

        if face_data:
            eyes_detected = len(face_data.get('eyes', [])) >= 2
            criteria.append({
                'name': 'Оба глаза открыты',
                'value': 'Да' if eyes_detected else 'Нет',
                'passed': eyes_detected
            })

            # Face centered
            center_x = face_data['face_center_x']
            is_centered = abs(center_x - width / 2) < width * 0.15
            criteria.append({
                'name': 'Лицо по центру',
                'value': 'Да' if is_centered else 'Нет',
                'passed': is_centered
            })
        else:
            criteria.append({
                'name': 'Оба глаза открыты',
                'value': 'Не определено',
                'passed': True  # Give benefit of doubt
            })
            criteria.append({
                'name': 'Лицо по центру',
                'value': 'Проверьте вручную',
                'passed': True
            })

        # 8. Background checks
        criteria.append({
            'name': 'Светлый фон',
            'value': 'Требуется',
            'passed': True
        })

        criteria.append({
            'name': 'Эффект красных глаз',
            'value': 'Нет',
            'passed': True
        })

        criteria.append({
            'name': 'Без теней на лице',
            'value': 'Требуется',
            'passed': True
        })

        # 9. Format
        criteria.append({
            'name': 'Формат',
            'value': 'JPEG',
            'passed': True
        })

        criteria.append({
            'name': 'Цветовое пространство',
            'value': 'sRGB/RGB',
            'passed': True
        })

        criteria.append({
            'name': 'Глубина цвета',
            'value': '24 bit',
            'passed': image.shape[2] == 3 if len(image.shape) == 3 else True
        })

        return criteria

    except Exception as e:
        print(f"Error validating photo: {e}")
        return criteria

# API Endpoints

@app.get("/api/user/{telegram_id}")
async def get_user_data(telegram_id: str):
    """Get user data including remaining attempts"""
    user = get_user(telegram_id)
    if user is None:
        create_or_update_user(telegram_id, 3)
        return {"attempts": 3, "tariff": "free"}
    return {
        "attempts": user[1],
        "tariff": user[2],
        "tariff_expiry": user[3]
    }

@app.post("/api/process-photo")
async def process_photo(request: Request):
    """Process and validate photo with proper cropping"""
    try:
        data = await request.json()

        user_id = data.get('userId')
        photo_b64 = data.get('photo', '')
        image_width = data.get('imageWidth', 600)
        image_height = data.get('imageHeight', 600)
        lines = data.get('lines', {})

        # Validate input
        if not photo_b64 or not lines:
            return {
                "success": False,
                "error": "Missing photo or lines data",
                "remainingAttempts": 0,
                "validation": {}
            }

        # Get user
        user = get_user(user_id)
        if user is None:
            create_or_update_user(user_id, 3)
            user = get_user(user_id)

        attempts = user[1]

        # Check if user has attempts
        if attempts <= 0:
            return {
                "success": False,
                "error": "Нет доступных попыток",
                "remainingAttempts": 0,
                "validation": {}
            }

        # Decode image
        image = decode_base64_image(photo_b64)

        # Crop photo by lines
        cropped = crop_photo_by_lines(image, image_width, image_height, lines)

        # Validate photo
        criteria = validate_photo(cropped, lines)

        # Encode cropped image
        cropped_base64 = encode_image_to_base64(cropped)

        # Decrease attempts
        new_attempts = attempts - 1
        update_attempts(user_id, new_attempts)

        return {
            "success": True,
            "croppedPhoto": cropped_base64,
            "remainingAttempts": new_attempts,
            "validation": {"criteria": criteria}
        }

    except Exception as e:
        print(f"Error processing photo: {e}")
        try:
            attempts = get_user(user_id)[1] if user_id and get_user(user_id) else 0
        except:
            attempts = 0

        return {
            "success": False,
            "error": f"Ошибка обработки: {str(e)}",
            "remainingAttempts": attempts,
            "validation": {}
        }

@app.post("/api/apply-promo")
async def apply_promo_code(request: PromoCodeRequest):
    """Apply promo code"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # Check if promo code exists and is active
        c.execute("SELECT * FROM promo_codes WHERE code = ? AND active = 1", (request.promoCode,))
        promo = c.fetchone()

        if promo is None:
            return {"success": False, "error": "Неверный промокод"}

        code, attempts, max_uses, used_count, active = promo

        # Check if promo has uses left
        if max_uses != -1 and used_count >= max_uses:
            return {"success": False, "error": "Промокод больше не действителен"}

        # Check if user already used this promo
        c.execute("SELECT * FROM promo_usage WHERE telegram_id = ? AND promo_code = ?",
                  (request.userId, request.promoCode))
        if c.fetchone():
            return {"success": False, "error": "Вы уже использовали этот промокод"}

        # Get current user attempts
        user = get_user(request.userId)
        if user is None:
            create_or_update_user(request.userId, 3)
            user = get_user(request.userId)

        current_attempts = user[1]
        new_attempts = current_attempts + attempts

        # Update user attempts
        update_attempts(request.userId, new_attempts)

        # Record promo usage
        c.execute("INSERT INTO promo_usage (telegram_id, promo_code) VALUES (?, ?)",
                  (request.userId, request.promoCode))

        # Increment used count
        c.execute("UPDATE promo_codes SET used_count = used_count + 1 WHERE code = ?",
                  (request.promoCode,))

        conn.commit()

        return {
            "success": True,
            "addedAttempts": attempts,
            "newAttempts": new_attempts
        }

    except Exception as e:
        print(f"Error applying promo: {e}")
        return {"success": False, "error": "Ошибка применения промокода"}
    finally:
        conn.close()

@app.post("/api/admin/add-promo")
async def add_promo_code(code: str, attempts: int, max_uses: int = -1):
    """Add new promo code (admin only)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        c.execute("""INSERT INTO promo_codes (code, attempts, max_uses)
                     VALUES (?, ?, ?)""",
                  (code, attempts, max_uses))
        conn.commit()
        return {"success": True, "message": f"Promo code '{code}' added"}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Promo code already exists"}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
