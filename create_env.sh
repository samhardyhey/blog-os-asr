conda config --set auto_activate_base false

# echo "Creating p38 conda env"
# conda create -n p38 python=3.8 -y
# eval "$(conda shell.bash hook)"
# conda activate p38

echo "Installing project requirements"
pip install -r ./requirements.txt

echo "Install pyannote.audio"
pip install pyannote.audio==1.1.2

echo "Installing torch/conda binaries"
conda install pytorch torchvision -c pytorch -y

echo "Testing torch installation"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

echo "Add git config"
git config --global user.name "Sam Hardy"
git config --global user.email "samhardyhey@gmail.com"

echo "Clone/install ctc decode"
if [ -d "ctcdecode" ]; then
  cd ctcdecode && pip install .
else
  git clone --recursive https://github.com/parlance/ctcdecode.git && cd ctcdecode && pip install .
fi

# echo "Install nemo toolkit"
# pip install nemo_toolkit['all']

# echo "Installing low-level audio libraries"
# apt-get update -y
# apt-get install libsndfile1 -y
# apt install ffmpeg -y