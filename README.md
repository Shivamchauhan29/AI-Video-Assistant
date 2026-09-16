# AI Video Assistant

AI Video Assistant is a Streamlit and CLI project that turns a YouTube URL or local media file into a transcript, summary, action items, key decisions, open questions, and a chat interface over the meeting transcript.

## Features

- Download audio from a YouTube URL or convert a local video/audio file to WAV.
- Transcribe audio with Whisper for English or Sarvam for Hinglish.
- Generate a title, summary, action items, key decisions, and open questions.
- Build a retrieval-based chat layer over the transcript.
- Run as a Streamlit app or from the command line.

## Project Structure

- `app.py` - Streamlit UI.
- `main.py` - CLI entrypoint.
- `core/` - Transcription, summarization, extraction, RAG, and vector store logic.
- `utils/audio_processor.py` - Audio download, conversion, and chunking.
- `Requirements.txt` - Python dependencies.

## Requirements

- Conda environment named `ai-video`.
- Python 3.11 in that environment.
- FFmpeg installed separately and available on your PATH.
- API keys for external services when using those features:
  - `MISTRAL_API_KEY`
  - `SARVAM_API_KEY` for Hinglish transcription

## Setup

1. Activate the conda environment:

   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate ai-video
   ```
2. Install the project dependencies:

   ```bash
   python -m pip install -r Requirements.txt
   ```
3. Create a `.env` file if you need API keys:

   ```env
   MISTRAL_API_KEY=your_key_here
   SARVAM_API_KEY=your_key_here
   WHISPER_MODEL=small
   SARVAM_STT_MODEL=saaras:v2.5
   ```

## Run the App

### Streamlit UI

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-video
python -m streamlit run app.py --server.headless true --server.fileWatcherType none --server.port 8501
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

### CLI

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-video
python main.py
```

The CLI will prompt for a YouTube URL or local file path and a language choice.

## How It Works

1. `utils.audio_processor` downloads or converts input media and splits it into chunks.
2. `core.transcriber` transcribes each chunk.
3. `core.summarizer` generates the title and summary.
4. `core.extractor` extracts action items, decisions, and questions.
5. `core.rag_engine` builds a transcript-based chat chain.

## Troubleshooting

- If Streamlit complains about the port being in use, stop the old process and run it again.
- If you see missing module errors, re-run dependency installation inside the `ai-video` conda environment.
- If FFmpeg is missing, install it separately before running audio conversion or YouTube downloads.
- If transcription or RAG calls fail, check that the relevant API key is set in `.env`.

## Notes

- The project was verified to start from the `ai-video` conda environment.
- The Streamlit launcher uses lazy imports so the UI can start even when the heavier ML stack is not loaded yet.
