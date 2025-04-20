python3 -m venv .music_tagging_env
source .music_tagging_env/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)" >> .music_tagging_env/bin/activate

wandb login