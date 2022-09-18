## OS ASR
Notebooks and scripts for the transcription of audio using open-source tools and models. See the accompanying blog post [here](https://www.samhardyhey.com/poor-mans-asr-pt-2).

## Install
- Conda env creation, python dependencies, low-level audio tools via `create_env.sh`
- Note some of the models referenced within `2_os_asr.ipynb` and `asr.py` can be quite large

## Usage
- Retrieve the first page's worth of podcast audio/transcripts via `python retrieve_transcripts.py`
- Transcribe audio files via `python transcribe_audio_os.py <input_audio_file> <output_transcript_file.csv>`