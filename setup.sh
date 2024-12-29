cd ..
git clone https://github.com/565353780/base-trainer.git

cd base-trainer
./setup.sh

pip install -U fairscale fire blobfile opencv-python \
  timm scipy
