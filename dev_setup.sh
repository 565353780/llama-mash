cd ..
git clone git@github.com:565353780/base-trainer.git

cd base-trainer
./dev_setup.sh

pip install -U fairscale fire blobfile opencv-python \
  timm scipy
