# Dhvani AI Voice Assistant (Twilio + OpenAI Realtime)

Dhvani (Digital Helper for Voice Assisted Natural Interaction) is a real-time AI voice assistant system built using FastAPI, Twilio Voice, and the OpenAI Realtime API.

This project enables users to interact with an AI over a phone call. It supports:

* Real-time voice conversations
* AI-generated responses
* Outbound call handling via Twilio
* Retrieval-Augmented Generation (RAG) using PDF (LPG module)

---

## Features

### General Dhvani

* Real-time conversational AI over phone call
* Uses OpenAI Realtime API for streaming responses
* Handles live speech interruptions
* Multi-language support (responds in user's language)

### LPG Dhvani (RAG System)

* Answers queries using a PDF knowledge base
* Uses ChromaDB for vector storage
* Semantic search with embeddings
* Strict context-based answering (no hallucination)

---

## Project Structure

```id="nkmzbo"
.
├── General_dhvani.py     # Main FastAPI app (entry point)
├── lpg_dhvani.py         # LPG module (APIRouter)
├── LPG.pdf               # Knowledge base for RAG
├── .env                  # Environment variables
└── README.md
```

---

## Tech Stack

* FastAPI – Backend framework
* Twilio – Voice call handling
* OpenAI Realtime API – AI voice responses
* WebSockets – Real-time streaming
* ChromaDB – Vector database
* LangChain – Embeddings and document processing
* PyMuPDF – PDF parsing

---

## Environment Variables

Create a `.env` file in the root directory:

```env id="0nxae3"
OPENAI_API_KEY=your_openai_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
NGROKURL=https://your-ngrok-url
```

---

## How to Run

### 1. Install dependencies

```bash id="79m93i"
pip install fastapi uvicorn websockets python-dotenv twilio langchain chromadb pymupdf scipy numpy
```

---

### 2. Start FastAPI server

```bash id="ybby56"
uvicorn General_dhvani:app --host 0.0.0.0 --port 5000
```

---

### 3. Start ngrok

```bash id="dbpt8m"
ngrok http 5000
```

Copy the generated URL and update it in `.env` as:

```env id="62v3t6"
NGROKURL=https://xxxx.ngrok-free.app
```

---

## API Endpoints

### General AI Call

```id="nl32y0"
GET /General_Dhvani?to=PHONE_NUMBER
```

Initiates a normal AI voice call.

---

### LPG AI Call (PDF-based)

```id="lo1z7g"
GET /lpg/LPG_Dhvani?to=PHONE_NUMBER
```

Initiates a call with RAG-based AI (answers from PDF).

---

### Health Check

```id="fw4yze"
GET /
```

---

## WebSocket Endpoints

* `/media-stream` → General AI streaming
* `/lpg/media-stream` → LPG AI streaming

---

## How It Works

### General Flow

1. User initiates call via API
2. Twilio connects call to FastAPI
3. Audio stream sent via WebSocket
4. OpenAI Realtime processes audio
5. AI response streamed back to caller

---

### LPG Flow (RAG)

1. PDF is loaded and split into chunks
2. Embeddings stored in ChromaDB
3. User query converted to embedding
4. Top similar chunks retrieved
5. AI generates response using retrieved context

---

## Important Notes

* Only one FastAPI server is required
* Only one ngrok URL is needed
* Do not hardcode ngrok URLs; always use `.env`
* Ensure Twilio webhook URLs match ngrok URL

---

## Future Improvements

* Add call logging and analytics
* Improve latency handling
* Add multi-document support
* Deploy using Docker
* Add authentication layer

---

## Author

Easy AI Team
Led by Ajinkya Joshi

---

## Summary

This project demonstrates a powerful combination of:

* Real-time AI voice interaction
* Telephony integration
* Retrieval-based intelligence

It can be extended into:

* Customer support bots
* IVR automation
* Voice-based assistants

“Talk to AI. Over a phone call.”
