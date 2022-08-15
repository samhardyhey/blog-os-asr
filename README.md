## Otso STT
#### 1.0 Overview
Otso's maiden STT wrapper. Bundles three models (diarization, ASR, Punctuation). Noted that it will take a few minutes to download all models when instantiating for the first time.

Additionally, you'll probably need to install the following audio libraries with sudo privileges (permission denied/lock issues?):
`apt-get update && apt-get install -y libsndfile1 ffmpeg`

As well as ctcdecode (used for timestamp extraction):
`git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .`

#### 2.0 Inference
Instantiate, call:
```python
from otso_stt_wrapper.model import OtsoSTT
otso_stt = OtsoSTT()

input_file = 'path/to/audio/file.wav'
# main predict single method still under development
res = otso_stt._transcribe_channel_seperated_audio(input_file)