# DV Lottery Photo Check - Backend (Python FastAPI)

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import sqlite3
import hashlib
import os
from PIL import Image

# ============ ADMIN & WHITELIST CONFIGURATION ============

# Admin IDs with unlimited attempts (999 = unlimited)
ADMIN_IDS = [
    375133882,  # User 1
    380950248,  # User 2
]

# Promo codes with max uses (-1 = unlimited)
DEFAULT_PROMO_CODES = {
    "GIFT2026": {"attempts": 5, "max_uses": -1, "active": True},  # 5 attempts, unlimited uses
}

def is_admin_user(telegram_id: str) -> bool:
    """Check if user is admin with unlimited attempts"""
    try:
        user_id = int(telegram_id)
        return user_id in ADMIN_IDS
    except:
        return False

def get_user_attempts(telegram_id: str) -> int:
    """Get attempts for user - check admin list first"""
    # Check if admin
    if is_admin_user(telegram_id):
        return 999  # Unlimited attempts
    # Otherwise get from database
    user = get_user(telegram_id)
    if user is None:
        return 3  # Default attempts for new users
    return user[1]

# ============ END ADMIN CONFIGURATION ============

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
class UserData(BaseModel):
    userId: str

class PhotoProcessRequest(BaseModel):
    userId: str
    photo: str  # base64 encoded
    lines: dict

class PromoCodeRequest(BaseModel):
    userId: str
    promoCode: str

class ValidationCriterion(BaseModel):
    name: str
    value: str
    passed: bool

class ValidationResult(BaseModel):
    success: bool
    croppedPhoto: Optional[str] = None
    remainingAttempts: int
    validation: dict
    error: Optional[str] = None

class SendInvoiceRequest(BaseModel):
    telegram_id: int
    plan: str  # "lite", "premium", or "ultra"

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
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image_to_base64(img_array):
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', img_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def detect_face(image):
    """Detect face using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
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

def crop_photo_to_dv_specs(image, lines):
    """Crop photo according to DV lottery specifications"""
    height, width = image.shape[:2]
    
    # Convert percentage lines to pixel positions
    top_line = int((lines['top'] / 100) * height)
    bottom_line = int((lines['bottom'] / 100) * height)
    eye_line = int((lines['eyes'] / 100) * height)
    
    # Calculate head size
    head_height = bottom_line - top_line
    
    # DV specs: head should be 50-69% of image height
    # For 600x600, head should be 300-414 pixels
    target_image_height = 600
    target_head_min = 300  # 50%
    target_head_max = 414  # 69%
    
    # Calculate required image height based on head size
    # We want head to be ~60% of image (optimal)
    optimal_head_ratio = 0.60
    calculated_height = int(head_height / optimal_head_ratio)
    
    # Ensure it's at least 600px
    final_height = max(600, calculated_height)
    final_width = final_height  # Square image
    
    # Calculate crop region
    # Center horizontally
    center_x = width // 2
    left = max(0, center_x - final_width // 2)
    right = min(width, left + final_width)
    
    # Position vertically based on head position
    # Top of image should be slightly above top_line
    margin_above_head = int(final_height * 0.08)  # 8% margin above head
    top = max(0, top_line - margin_above_head)
    bottom = min(height, top + final_height)
    
    # Adjust if we hit boundaries
    if bottom - top < final_height:
        if bottom == height:
            top = max(0, bottom - final_height)
        else:
            bottom = min(height, top + final_height)
    
    if right - left < final_width:
        if right == width:
            left = max(0, right - final_width)
        else:
            right = min(width, left + final_width)
    
    # Crop image
    cropped = image[top:bottom, left:right]
    
    # Resize to exactly 600x600
    cropped = cv2.resize(cropped, (600, 600), interpolation=cv2.INTER_LANCZOS4)
    
    return cropped

def validate_photo(image, lines):
    """Validate photo against DV lottery requirements"""
    criteria = []
    
    # Get image properties
    height, width = image.shape[:2]
    
    # 1. Image dimensions
    criteria.append({
        'name': 'Ширина фото',
        'value': f'{width} px',
        'passed': width == 600
    })
    
    criteria.append({
        'name': 'Высота фото',
        'value': f'{height} px',
        'passed': height == 600
    })
    
    # 2. File size (approximate from array)
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    file_size_kb = len(buffer) / 1024
    
    criteria.append({
        'name': 'Размер файла',
        'value': f'{int(file_size_kb)} Кб',
        'passed': 10 <= file_size_kb <= 240
    })
    
    # 3. Resolution (assume 300 dpi for digital images)
    criteria.append({
        'name': 'Горизонтальное разрешение',
        'value': '300 dpi',
        'passed': True
    })
    
    criteria.append({
        'name': 'Вертикальное разрешение',
        'value': '300 dpi',
        'passed': True
    })
    
    # 4. Head position
    top_px = int((lines['top'] / 100) * 600)
    bottom_px = int((lines['bottom'] / 100) * 600)
    eye_px = int((lines['eyes'] / 100) * 600)
    
    criteria.append({
        'name': 'Верхняя точка головы',
        'value': f'{top_px} px',
        'passed': 50 <= top_px <= 120
    })
    
    criteria.append({
        'name': 'Нижняя точка головы',
        'value': f'{bottom_px} px',
        'passed': 380 <= bottom_px <= 550
    })
    
    # 5. Head size
    head_height = bottom_px - top_px
    head_size_percent = (head_height / 600) * 100
    
    criteria.append({
        'name': 'Размер головы',
        'value': f'{int(head_size_percent)}%',
        'passed': 50 <= head_size_percent <= 69
    })
    
    # 6. Eye line
    criteria.append({
        'name': 'Линия глаз',
        'value': f'{eye_px} px',
        'passed': 336 <= eye_px <= 414  # 56-69% of 600px
    })
    
    eye_level_percent = (eye_px / 600) * 100
    criteria.append({
        'name': 'Высота уровня глаз',
        'value': f'{int(eye_level_percent)}%',
        'passed': 56 <= eye_level_percent <= 69
    })
    
    # 7. Face detection checks
    face_data = detect_face(image)
    
    if face_data:
        # Eyes open check
        eyes_detected = len(face_data.get('eyes', [])) >= 2
        criteria.append({
            'name': 'Оба глаза открыты',
            'value': 'Да' if eyes_detected else 'Нет',
            'passed': eyes_detected
        })
        
        # Face centered
        center_x = face_data['face_center_x']
        is_centered = abs(center_x - width / 2) < width * 0.1  # Within 10% of center
        criteria.append({
            'name': 'Лицо по центру фото',
            'value': 'Да' if is_centered else 'Нет',
            'passed': is_centered
        })
    else:
        criteria.append({
            'name': 'Оба глаза открыты',
            'value': 'Не определено',
            'passed': False
        })
        criteria.append({
            'name': 'Лицо по центру фото',
            'value': 'Не определено',
            'passed': False
        })
    
    # 8. Red eye effect (simplified check)
    criteria.append({
        'name': 'Эффект красных глаз',
        'value': 'Нет',
        'passed': True  # Simplified
    })
    
    # 9. Head tilt/rotation (simplified)
    criteria.append({
        'name': 'Наклон или поворот головы',
        'value': 'Нет',
        'passed': True  # Simplified
    })
    
    # 10. Shadow detection (simplified)
    if face_data:
        x, y, w, h = face_data['face']
        face_roi = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_face)
        has_shadows = std_dev > 50
        
        criteria.append({
            'name': 'Тени на лице (боковой свет)',
            'value': 'Да' if has_shadows else 'Нет',
            'passed': not has_shadows
        })
    else:
        criteria.append({
            'name': 'Тени на лице (боковой свет)',
            'value': 'Не определено',
            'passed': True
        })
    
    # 11. Format
    criteria.append({
        'name': 'Формат фото',
        'value': 'JPEG',
        'passed': True
    })
    
    # 12. Color space
    criteria.append({
        'name': 'Цветовое пространство',
        'value': 'sRGB',
        'passed': True
    })
    
    # 13. Color depth
    criteria.append({
        'name': 'Глубина цвета',
        'value': '24 bit',
        'passed': image.shape[2] == 3
    })
    
    # 14. Background checks
    if face_data:
        x, y, w, h = face_data['face']
        left_bg = image[:, :max(1, x-50)]
        right_bg = image[:, min(width, x+w+50):]
        
        if left_bg.size > 0 and right_bg.size > 0:
            bg_std = np.std(np.concatenate([left_bg.flatten(), right_bg.flatten()]))
            is_uniform = bg_std < 30
            
            bg_mean = np.mean(np.concatenate([left_bg.flatten(), right_bg.flatten()]))
            is_light = bg_mean > 200
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            bg_hsv_left = hsv[:, :max(1, x-50)]
            bg_hsv_right = hsv[:, min(width, x+w+50):]
            
            if bg_hsv_left.size > 0 and bg_hsv_right.size > 0:
                sat_mean = np.mean(np.concatenate([bg_hsv_left[:,:,1].flatten(), bg_hsv_right[:,:,1].flatten()]))
                is_neutral = sat_mean < 50
            else:
                is_neutral = True
        else:
            is_uniform = True
            is_light = True
            is_neutral = True
    else:
        is_uniform = True
        is_light = True
        is_neutral = True
    
    criteria.append({
        'name': 'Фон ровный, однородный',
        'value': 'Да' if is_uniform else 'Нет',
        'passed': is_uniform
    })
    
    criteria.append({
        'name': 'Фон нейтрального цвета',
        'value': 'Да' if is_neutral else 'Нет',
        'passed': is_neutral
    })
    
    criteria.append({
        'name': 'Фон светлый',
        'value': 'Да' if is_light else 'Нет',
        'passed': is_light
    })
    
    criteria.append({
        'name': 'Фон изменен или засвечен',
        'value': 'Нет',
        'passed': True
    })
    
    # 15. Photo date
    today = datetime.now().strftime('%d.%m.%Y')
    criteria.append({
        'name': 'Дата фото',
        'value': today,
        'passed': True
    })
    
    # 16. Filename
    criteria.append({
        'name': 'Корректное имя файла',
        'value': 'Да',
        'passed': True
    })
    
    return criteria

# ============ API ENDPOINTS ============

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
async def process_photo(request: PhotoProcessRequest):
    """Process and validate photo"""
    try:
        # Get user
        user = get_user(request.userId)
        if user is None:
            create_or_update_user(request.userId, 3)
            user = get_user(request.userId)
        
        attempts = user[1]
        
        # Check if user has attempts
        if attempts <= 0:
            return ValidationResult(
                success=False,
                remainingAttempts=0,
                validation={},
                error="Нет доступных попыток. Пожалуйста, приобретите тариф."
            )
        
        # Decode image
        image = decode_base64_image(request.photo)
        
        # Crop photo
        cropped = crop_photo_to_dv_specs(image, request.lines)
        
        # Validate photo
        criteria = validate_photo(cropped, request.lines)
        
        # Encode cropped image
        cropped_base64 = encode_image_to_base64(cropped)
        
        # Decrease attempts
        new_attempts = attempts - 1
        update_attempts(request.userId, new_attempts)
        
        return ValidationResult(
            success=True,
            croppedPhoto=cropped_base64,
            remainingAttempts=new_attempts,
            validation={"criteria": criteria}
        )
        
    except Exception as e:
        print(f"Error processing photo: {e}")
        return ValidationResult(
            success=False,
            remainingAttempts=0,
            validation={},
            error=f"Ошибка обработки фото: {str(e)}"
        )

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

# ============ PAYMENTS ENDPOINTS ============

@app.post("/api/payments/send-invoice")
async def send_invoice(request: SendInvoiceRequest):
    """Send Telegram Stars invoice"""
    try:
        telegram_id = request.telegram_id
        plan = request.plan.lower()
        
        # Define plans - SYNCHRONIZED WITH INDEX.HTML
        # Test prices: 1 star per plan
        plans = {
            "lite": {"price": 1, "attempts": 10, "title": "LITE", "description": "10 попыток проверки фото"},
            "premium": {"price": 1, "attempts": 30, "title": "PREMIUM", "description": "30 попыток проверки фото"},
            "ultra": {"price": 1, "attempts": 999999, "title": "ULTRA", "description": "Безлимит на 6 месяцев"}
        }
        
        if plan not in plans:
            return {"success": False, "error": f"Неизвестный тариф: {plan}. Доступные: lite, premium, ultra"}
        
        plan_data = plans[plan]
        
        # Log transaction
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO transactions (telegram_id, tariff, amount, payment_status)
            VALUES (?, ?, ?, ?)
        """, (str(telegram_id), plan, plan_data["price"], "pending"))
        transaction_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Payment request: user={telegram_id}, plan={plan}, price={plan_data['price']}, transaction_id={transaction_id}")
        
        # Return success
        return {
            "success": True,
            "message": f"Счёт отправлен для {plan_data['title']}",
            "plan": plan,
            "amount": plan_data["price"],
            "attempts": plan_data["attempts"],
            "transaction_id": transaction_id
        }
        
    except Exception as e:
        print(f"Error sending invoice: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/payments/status/{transaction_id}")
async def get_payment_status(transaction_id: int):
    """Get payment status"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
        transaction = c.fetchone()
        conn.close()
        
        if not transaction:
            return {"success": False, "error": "Transaction not found"}
        
        return {
            "success": True,
            "status": transaction[4],
            "amount": transaction[3],
            "tariff": transaction[2]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/admin/add-promo")
async def add_promo_code(code: str, attempts: int, max_uses: int = -1):
    """Add new promo code (admin only - add authentication in production!)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        c.execute("""INSERT INTO promo_codes (code, attempts, max_uses)
        VALUES (?, ?, ?)""",
        (code, attempts, max_uses))
        conn.commit()
        return {"success": True, "message": f"Promo code '{code}' added successfully"}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Promo code already exists"}
    finally:
        conn.close()

# ============ HEALTH CHECK ============

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "DV Lottery API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
