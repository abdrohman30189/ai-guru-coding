import sqlite3
import uuid
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# LIBRARY BARU: Untuk pencarian internet real-time
from duckduckgo_search import DDGS 

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
# 2. FUNGSI PENCARIAN (UPDATE BARU)
# ======================
def search_web(query: str) -> str:
    """Mencari informasi terbaru di internet menggunakan DuckDuckGo."""
    print(f"🔎 Sedang mencari: {query}...") # Info di terminal
    try:
        # Mengambil 3 hasil teratas agar tidak terlalu banyak token
        results = DDGS().text(keywords=query, region='id-id', max_results=3)
        if not results:
            return ""
        
        # Format hasil pencarian menjadi teks rapi
        formatted_results = "FAKTA TERBARU DARI INTERNET:\n"
        for i, res in enumerate(results, 1):
            formatted_results += f"{i}. {res['title']}: {res['body']}\n"
            
        return formatted_results
    except Exception as e:
        print(f"Search Error: {e}")
        return ""

# ======================
# 3. DATABASE SYSTEM (SQLite)
# ======================
def init_db():
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
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
              (session_id, role, content))
    conn.commit()
    conn.close()

def get_history(session_id: str) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    result: List[Dict[str, str]] = []
    for row in rows:
        result.append({"role": str(row["role"]), "content": str(row["content"])})
    return result

init_db()

# ======================
# 4. ROUTE: HOME
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = request.cookies.get("session_id")
    is_new_user = False
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_user = True

    chat_history = get_history(session_id) 

    response = templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history
    })

    if is_new_user:
        response.set_cookie(key="session_id", value=session_id, max_age=31536000)

    return response


# ======================
# 5. ROUTE: API CHAT (LOGIKA UTAMA)
# ======================
@app.post("/api/chat")
async def api_chat(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session expired"}, status_code=400)

    data = await request.json()
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # A. Simpan pesan User
    save_message(session_id, "user", user_msg)

    # B. LAKUKAN PENCARIAN INTERNET OTOMATIS
    # Kita cari info terkait pesan user agar AI tahu konteks terbaru
    web_context = search_web(user_msg)

    # C. Siapkan System Prompt yang Dinamis
    system_content = "Kamu adalah asisten AI yang cerdas dan membantu."
    
    # Jika ada hasil pencarian, kita paksa AI membacanya
    if web_context:
        system_content += f"\n\n[PENTING] Gunakan data berikut untuk menjawab pertanyaan pengguna secara akurat dan realtime:\n{web_context}"
    else:
        system_content += "\nJawablah berdasarkan pengetahuanmu sendiri."

    # D. Susun Pesan untuk OpenAI
    messages_payload: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_content}
    ]
    
    # Masukkan history chat (agar nyambung)
    db_history = get_history(session_id)[-6:] # Ambil 6 pesan terakhir saja biar cepat
    for msg in db_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            messages_payload.append({"role": "user", "content": content})
        elif role == "assistant":
            messages_payload.append({"role": "assistant", "content": content})

    try:
        # E. Panggil OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_payload,
            max_tokens=800 # Token ditambah agar jawaban lebih lengkap
        )

        bot_reply = response.choices[0].message.content or ""

        # F. Simpan pesan Bot
        save_message(session_id, "assistant", bot_reply)

        return JSONResponse({"reply": bot_reply})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
