conda config --set auto_activate_base false

echo "Creating p38 conda env"
conda create -n p38 python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate p38

echo "Install pyannote.audio"
# bypass hard torch pins, install first; yikes
pip install pyannote.audio==2.0.1
pip install pyannote.pipeline==2.3

echo "Install nemo toolkit"
pip install nemo_toolkit['all']

echo "Installing project requirements"
pip install -r ./requirements.txt

echo "Installing torch/conda binaries"
conda remove pytorch torchvision torchaudio -y # silly jarvis env
conda install pytorch torchvision torchaudio -c pytorch -y

echo "Testing torch installation"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

echo "Correct numpy version, truly in dependency hell now"
pip install numpy==1.22

echo "Add git config"
git config --global user.name "Sam Hardy"
git config --global user.email "samhardyhey@gmail.com"

echo "Installing low-level audio libraries"
apt-get update -y
apt-get install libsndfile1 -y
apt-get install libstdc++6 -y
apt install ffmpeg -y