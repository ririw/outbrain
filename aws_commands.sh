sudo apt-get install -y python3-pip python3-numpy cython3 awscli htop
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL sklearn xgboost pandas plumbum

