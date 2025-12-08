import sqlite3
import uuid
import os
from typing import List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ======================
# 1. KONFIGURASI
# ======================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

client = OpenAI(api_key=API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
DB_NAME = "chat_history.db"

# ======================
# 2. DATABASE SYSTEM (SQLite)
# ======================
def init_db():
    """Membuat tabel database jika belum ada."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    """Menyimpan pesan ke database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
              (session_id, role, content))
    conn.commit()
    conn.close()

def get_history(session_id: str) -> List[dict]:
    """Mengambil riwayat chat berdasarkan Session ID."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Agar hasil bisa diakses seperti dict
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    # Konversi ke format list of dict
    return [{"role": row["role"], "content": row["content"]} for row in rows]

# Jalankan inisialisasi DB saat server nyala
init_db()

# ======================
# 3. ROUTE: HOME (Frontend)
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Cek apakah user sudah punya Session ID (Cookie)
    session_id = request.cookies.get("session_id")
    
    # Jika belum punya, kita buatkan ID baru (tapi belum diset ke cookie browser di sini)
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_user = True
    else:
        is_new_user = False

    # Ambil riwayat chat lama dari Database
    chat_history = get_history(session_id)

    # Render HTML dengan data history
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history  # Kirim data ke frontend
    })

    # Jika user baru, tempelkan "stempel" cookie agar dia dikenali kedepannya
    if is_new_user:
        response.set_cookie(key="session_id", value=session_id, max_age=31536000) # Expire 1 tahun

    return response


# ======================
# 4. ROUTE: API CHAT
# ======================
@app.post("/api/chat")
async def api_chat(request: Request):
    # Ambil session_id dari cookie pengirim
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session expired, please refresh page."}, status_code=400)

    data = await request.json()
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # A. Simpan pesan User ke DB
    save_message(session_id, "user", user_msg)

    # B. Siapkan Context untuk OpenAI (Ambil history agar bot ingat)
    # Kita ambil max 10 pesan terakhir agar hemat token
    db_history = get_history(session_id)[-10:] 
    
    messages_payload = [{"role": "system", "content": "You are a helpful assistant."}]
    messages_payload.extend(db_history)

    try:
        # C. Panggil OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_payload, # type: ignore
            max_tokens=500
        )

        bot_reply = response.choices[0].message.content or ""

        # D. Simpan pesan Bot ke DB
        save_message(session_id, "assistant", bot_reply)

        return JSONResponse({"reply": bot_reply})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)