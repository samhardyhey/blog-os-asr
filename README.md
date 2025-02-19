# Open Source ASR 🎤

Audio transcription toolkit using open-source ASR models and tools. Companion code for ["Poor Man's ASR"](https://www.samhardyhey.com/poor-mans-asr-pt-1) and its [follow-up](https://www.samhardyhey.com/poor-mans-asr-pt-2).

## Features
- 🎧 Podcast audio retrieval
- 📝 Automated transcription
- 🤖 Open-source ASR models
- 📊 Performance evaluation

## Setup
```bash
# Install dependencies and audio tools
./create_env.sh
```

## Usage
```bash
# Download podcast samples
python retrieve_transcripts.py

# Generate transcriptions
python transcribe_audio_os.py input.mp3 output.csv
```

## Structure
- 📓 `2_os_asr.ipynb` # Main ASR notebook
- 🎵 `retrieve_transcripts.py` # Audio collection
- 🗣️ `transcribe_audio_os.py` # Transcription tool
- ⚙️ `create_env.sh` # Environment setup

*Note: Models require significant storage space. See notebook for details.*