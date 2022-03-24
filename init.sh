pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir data/
mkdir log/
mkdir results/
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
tar -xf nsynth-train.jsonwav.tar.gz
mv nsynth-train/audio/* data/
rm -r nsynth-train/
rm nsynth-train.jsonwav.tar.gz