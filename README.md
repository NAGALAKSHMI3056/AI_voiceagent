# AI Math Teacher Voice Agent

An interactive, deployable voice agent that teaches math, delivers news headlines, and reports current weather. Built for clarity, modularity, and real-time performance.

---

## Features

- Persona-driven math tutoring with step-by-step explanations  
- Real-time transcription via AssemblyAI  
- Location-aware news headlines (NewsAPI with RSS fallback)  
- Current weather reports (WeatherAPI.com)  
- Natural speech output using Murf TTS or browser synthesis  
- Per-session API key overrides in the UI—no `.env` edits required  

---

## Technology Stack

- **Backend**: FastAPI, Uvicorn  
- **Transcription**: AssemblyAI SDK  
- **Language Model**: Gemini API  
- **Text-to-Speech**: Murf TTS (with browser fallback)  
- **News & Weather**: NewsAPI.org, Google RSS, WeatherAPI.com  
- **Frontend**: HTML/CSS/JavaScript, MediaRecorder API  
- **Deployment**: Render.com (free tier with auto-deploy)  

---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/NAGALAKSHMI3056/AI_voiceagent
   cd ai-math-teacher-voice-agent
   ```

2. Create and activate a virtual environment  
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   .\venv\Scripts\activate     # Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. (Optional) Copy `.env.example` to `.env` and set default API keys.  
2. In the app UI, click **Settings** to enter or override keys for:  
   - AssemblyAI  
   - Murf  
   - Gemini  
   - NewsAPI  
   - WeatherAPI  

Overrides persist in browser `localStorage` and server memory.

---

## Running Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your browser at `http://localhost:8000` to begin.

---

## API Endpoints

- `GET  /`  
  Serves the chat UI.

- `GET  /api/config/{session_id}`  
  Returns default or overridden API keys for a session.

- `POST /api/config/{session_id}`  
  Accepts JSON payload of API key overrides.

- `POST /agent/chat/{session_id}`  
  Handles audio upload (`multipart/form-data`), optional `flow=news` query.  
  Response: `{ user_text, llm_text, audio_url }`.

---

## Deployment on Render

1. Push to GitHub.  
2. In Render Dashboard, create a **Web Service**:  
   - **Build Command**: `pip install -r requirements.txt`  
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`  
3. Add environment variables or use `render.yaml`.  
4. Enable auto-deploy on your main branch.

---

## Contributing

Contributions and feedback are welcome. Please open an issue or pull request to propose enhancements.

---


## Live Demo

Experience the AI Math Teacher voice agent in action:  
https://ai-voiceagent.onrender.com
Just open the link in your browser to start interacting.

---
