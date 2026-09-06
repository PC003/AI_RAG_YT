# YouTube Playlist RAG

A complete end-to-end system for running a Retrieval-Augmented Generation (RAG) pipeline over YouTube playlists using local Whisper transcription and local embeddings.

## 🏗 Architecture

```
flowchart TD
    A[YouTube Playlist URL] --> B[yt-dlp: Extract Metadata]
    B --> C[yt-dlp: Download Audio]
    C --> D[Local Whisper (Transcribe)]
    D --> E[Timestamp-Aware Chunking]
    E --> F[all-MiniLM-L6-v2 Embeddings]
    F --> G[(Qdrant Vector DB)]
    
    U[User Query] --> H[all-MiniLM-L6-v2 Embeddings]
    H --> I[Qdrant Similarity Search]
    I --> G
    I --> J[Top-K Chunks]
    J --> K[LLM]
    K --> L[Answer + YouTube Timestamp Links]
```

## 📋 Prerequisites

- **Python:** `3.11+`
- **System Packages:**
  - `ffmpeg` (required by Whisper and yt-dlp)

On macOS:
```bash
brew install ffmpeg
```

On Ubuntu/Debian:
```bash
sudo apt update && sudo apt install ffmpeg
```

## 🚀 Setup

1. **Clone & Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration:**
   Copy the example config:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` as needed.

3. **Start Qdrant with Docker:**
   ```bash
   docker compose up -d qdrant
   ```

   Qdrant stores data persistently in `data/qdrant`, so recreating the
   container will not delete your indexed playlists.

4. **Install Ollama (Optional, default):**
   If you use the default `ollama` provider for the LLM step:
   - Download [Ollama](https://ollama.ai/)
   - Run `ollama pull llama3`

## ⚙️ Running the App

Start the Streamlit UI:
```bash
streamlit run app/ui.py
```

1. Go to the **Ingest Playlist** tab and paste a YouTube Playlist URL. Watch the progress bar as it downloads, transcribes, embeds, and indexes. Large playlists take time.
2. Go to the **Chat** tab, choose **All playlists** or one playlist, and ask questions. You will get grounded answers and clickable YouTube links that open the video at the exact timestamp where the context was discussed.

## 🧪 Testing

Run unit tests via:
```bash
pytest tests/
```
