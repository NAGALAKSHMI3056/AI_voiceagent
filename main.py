import os
import re
import shutil
import tempfile
import logging
import xml.etree.ElementTree as ET

from pathlib import Path as FSPath
from dotenv import load_dotenv

import httpx
import assemblyai as aai

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Path as ApiPath,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Load .env (optional defaults) ----------
load_dotenv()

ENV_ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
ENV_MURF_KEY       = os.getenv("MURF_API_KEY",       "").strip()
ENV_GEMINI_KEY     = os.getenv("GEMINI_API_KEY",     "").strip()
ENV_NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY",        "").strip()
ENV_WEATHER_KEY    = os.getenv("WEATHER_API_KEY",    "").strip()
GEMINI_MODEL       = os.getenv("GEMINI_MODEL",       "gemini-2.0-flash").strip()
DEFAULT_CITY       = os.getenv("DEFAULT_CITY",       "Nellore").strip()

missing = [
    name for name, val in [
        ("ASSEMBLYAI_API_KEY", ENV_ASSEMBLYAI_KEY),
        ("MURF_API_KEY",       ENV_MURF_KEY),
        ("GEMINI_API_KEY",     ENV_GEMINI_KEY),
        ("NEWSAPI_KEY",        ENV_NEWSAPI_KEY),
        ("WEATHER_API_KEY",    ENV_WEATHER_KEY),
    ] if not val
]
if missing:
    logger.warning(
        f"Environment defaults missing for: {missing}. "
        "Users must supply keys via the UI Settings modal."
    )

# ---------- AssemblyAI default setup ----------
aai.settings.api_key = ENV_ASSEMBLYAI_KEY

# ---------- App & Static Paths ----------
UPLOADS_DIR = FSPath("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR = FSPath("static")
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def log_config_keys():
    """
    Log the first and last few characters of each key on startup.
    Helps confirm the service sees the correct values.
    """
    def short(k: str) -> str:
        return f"{k[:4]}…{k[-4:]}" if len(k) > 8 else k or "(empty)"
    logger.info("Loaded keys – "
                f"AssemblyAI: {short(ENV_ASSEMBLYAI_KEY)}, "
                f"Murf: {short(ENV_MURF_KEY)}, "
                f"Gemini: {short(ENV_GEMINI_KEY)}, "
                f"NewsAPI: {short(ENV_NEWSAPI_KEY)}, "
                f"WeatherAPI: {short(ENV_WEATHER_KEY)}"
                )


# ---------- In-Memory Config Store ----------
# session_id → { assembly_ai, murf, gemini, newsapi, weatherapi }
config_store: dict[str, dict] = {}


@app.get("/api/config/{session_id}")
async def get_config(session_id: str):
    overrides = config_store.get(session_id, {})
    return {
        "assembly_ai": overrides.get("assembly_ai", ENV_ASSEMBLYAI_KEY),
        "murf":        overrides.get("murf",        ENV_MURF_KEY),
        "gemini":      overrides.get("gemini",      ENV_GEMINI_KEY),
        "newsapi":     overrides.get("newsapi",     ENV_NEWSAPI_KEY),
        "weatherapi":  overrides.get("weatherapi",  ENV_WEATHER_KEY),
    }


@app.post("/api/config/{session_id}")
async def set_config(session_id: str, payload: dict):
    """
    Accepts JSON payload with any subset of API keys.
    Non-empty keys overwrite the defaults for that session.
    """
    session_cfg = config_store.setdefault(session_id, {})
    for key in ("assembly_ai", "murf", "gemini", "newsapi", "weatherapi"):
        if payload.get(key):
            session_cfg[key] = payload[key].strip()
    return {"status": "ok", "overrides": session_cfg}


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """
    Serve the chat UI for both GET and HEAD requests.
    """
    return FileResponse(STATIC_DIR / "index.html")


# ---------- Chat History Helpers ----------
chat_history_store: dict[str, list[dict]] = {}


def get_chat_history(sid: str) -> list[dict]:
    return chat_history_store.get(sid, [])


def add_message(sid: str, role: str, text: str):
    chat_history_store.setdefault(sid, []).append({"role": role, "text": text})


def build_gemini_contents(sid: str) -> list[dict]:
    SYSTEM_PROMPT = (
        "You are a patient, enthusiastic math teacher. "
        "Explain each step clearly, ask guiding questions, "
        "and offer positive reinforcement."
    )
    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
    for m in get_chat_history(sid):
        contents.append({"role": m["role"], "parts": [{"text": m["text"]}]})
    return contents


# ---------- News & Weather Helpers ----------

def sanitize_location(loc: str) -> str:
    return re.sub(r"[^\w\s]", "", loc).strip().lower()


def map_location_to_country(loc: str) -> str:
    mapping = {
        "india": "in", "united states": "us", "usa": "us",
        "uk": "gb", "united kingdom": "gb", "germany": "de",
        "france": "fr", "canada": "ca", "australia": "au"
    }
    return mapping.get(sanitize_location(loc), "in")

async def fetch_latest_news(country: str, limit: int = 5) -> list[dict]:
    # Pick override or env key
    sess_key = None
    for cfg in config_store.values():
        if "newsapi" in cfg:
            sess_key = cfg["newsapi"]
            break
    api_key = sess_key or ENV_NEWSAPI_KEY

    logger.info(f"[News Flow] using API key: {api_key[:4]}…{api_key[-4:]} for country '{country}'")

    if api_key:
        try:
            params = {"apiKey": api_key, "country": country, "pageSize": limit}
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://newsapi.org/v2/top-headlines", params=params)
            logger.info(f"[News Flow] HTTP {resp.status_code} → {resp.text[:200]}")
            data = resp.json()
            if data.get("status") == "ok":
                return data.get("articles", [])
            logger.warning(f"[News Flow] NewsAPI error: {data.get('message')}")
        except Exception as e:
            logger.warning(f"[News Flow] exception: {e}; falling back to RSS")
    # … RSS fallback unchanged …
    # RSS fallback
    feed_url = (
        f"https://news.google.com/rss?"
        f"hl=en-{country.upper()}&gl={country.upper()}&ceid={country.upper()}:en"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(feed_url)
        resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:limit]
    return [
        {
            "title": it.findtext("title", "No title"),
            "url":   it.findtext("link", "#"),
            "source": (it.find("source") or ET.Element("")).text or ""
        }
        for it in items
    ]


async def get_weather(city: str, api_key: str) -> dict:
    if not api_key:
        raise HTTPException(400, detail="WeatherAPI key not configured.")
    url = "https://api.weatherapi.com/v1/current.json"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params={"key": api_key, "q": city, "aqi": "no"})
    if resp.status_code == 401:
        logger.error(f"WeatherAPI.com 401 Unauthorized → {resp.text}")
        raise HTTPException(503, detail="Weather service unavailable.")
    resp.raise_for_status()
    data = resp.json()["current"]
    return {
        "temp":       data["temp_c"],
        "desc":       data["condition"]["text"],
        "humidity":   data["humidity"],
        "wind_speed": data["wind_kph"] / 3.6,
    }


# ---------- Unified Chat Endpoint ----------

@app.post("/agent/chat/{session_id}")
async def chat_with_history(
    session_id: str = ApiPath(...),
    file: UploadFile   = File(...),
    flow: str          = Query(None)
):
    # Load per-session or environment keys
    ov             = config_store.get(session_id, {})
    assembly_key   = ov.get("assembly_ai",  ENV_ASSEMBLYAI_KEY)
    murf_key       = ov.get("murf",         ENV_MURF_KEY)
    gemini_key     = ov.get("gemini",       ENV_GEMINI_KEY)
    newsapi_key    = ov.get("newsapi",      ENV_NEWSAPI_KEY)
    weatherapi_key = ov.get("weatherapi",   ENV_WEATHER_KEY)

    if not assembly_key:
        raise HTTPException(400, detail="AssemblyAI API key not configured.")

    # Transcribe with AssemblyAI
    aai.settings.api_key = assembly_key
    transcriber = aai.Transcriber()

    with tempfile.NamedTemporaryFile(
        dir=UPLOADS_DIR, suffix=".wav", delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        wav_path = tmp.name

    try:
        tx = transcriber.transcribe(wav_path)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, detail=str(e))
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    if tx.error or not tx.text.strip():
        raise HTTPException(400, detail=tx.error or "Empty transcript")

    user_text = tx.text.strip()
    add_message(session_id, "user", user_text)

    # ----- News Flow -----
    if flow == "news":
        country = map_location_to_country(user_text)
        articles = await fetch_latest_news(country)

        if not articles:
            llm_txt = f"No news available for {user_text}."
        else:
            display = "\n".join(
                f"{i+1}. {a['title']} ({a['source']}) — {a['url']}"
                for i, a in enumerate(articles)
            )
            llm_txt = f"Top headlines for {user_text}:\n{display}"

        add_message(session_id, "model", llm_txt)

        audio_url = None
        if murf_key and articles:
            try:
                tts_text = " ".join(
                    f"{i+1}. {a['title']} from {a['source']}"
                    for i, a in enumerate(articles)
                )
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.murf.ai/v1/speech/generate",
                        json={
                            "text": tts_text,
                            "voice_id": "en-IN-alia",
                            "audio_format": "mp3"
                        },
                        headers={"api-key": murf_key}
                    )
                    resp.raise_for_status()
                    audio_url = resp.json().get("audioFile") or resp.json().get("audio_url")
            except Exception as e:
                logger.warning(f"Murf TTS failed for news: {e}")

        return {"user_text": user_text, "llm_text": llm_txt, "audio_url": audio_url}

    # ----- Weather Flow -----
    if "weather" in user_text.lower():
        match = re.search(r"weather in ([\w\s]+)", user_text.lower())
        city = (match.group(1).strip() if match else DEFAULT_CITY).title()

        try:
            weather = await get_weather(city, weatherapi_key)
            display = (
                f"Weather in {city}: {weather['desc']}, {weather['temp']}°C. "
                f"Humidity: {weather['humidity']}%. Wind: {weather['wind_speed']:.1f} m/s."
            )
        except HTTPException as he:
            display = he.detail
        except Exception:
            display = f"Weather data for {city} is unavailable."

        add_message(session_id, "model", display)

        audio_url = None
        if murf_key:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.murf.ai/v1/speech/generate",
                        json={"text": display, "voice_id": "en-IN-alia", "audio_format": "mp3"},
                        headers={"api-key": murf_key}
                    )
                    resp.raise_for_status()
                    audio_url = resp.json().get("audioFile") or resp.json().get("audio_url")
            except Exception as e:
                logger.warning(f"Murf TTS failed for weather: {e}")

        return {"user_text": user_text, "llm_text": display, "audio_url": audio_url}

    # ----- General Queries via Gemini -----
    if not gemini_key:
        raise HTTPException(400, detail="Gemini API key not configured.")

    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            gemini_url,
            json={"contents": build_gemini_contents(session_id)},
            headers={"Content-Type": "application/json"},
            params={"key": gemini_key}
        )
        raw = await resp.aread()

        if resp.status_code != 200:
            logger.error(raw.decode(errors="ignore"))
            raise HTTPException(500, detail="Gemini API error")

        data = resp.json()

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    reply = parts[0].get("text", "").strip() if parts else ""
    add_message(session_id, "model", reply)

    audio_url = None
    if murf_key:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.murf.ai/v1/speech/generate",
                    json={"text": reply, "voice_id": "en-IN-alia", "audio_format": "mp3"},
                    headers={"api-key": murf_key}
                )
                resp.raise_for_status()
                audio_url = resp.json().get("audioFile") or resp.json().get("audio_url")
        except Exception as e:
            logger.warning(f"Murf TTS failed for Gemini reply: {e}")

    return {"user_text": user_text, "llm_text": reply, "audio_url": audio_url}
