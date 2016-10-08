export HOST=54.252.63.12
sudo locale-gen en_AU.UTF-8
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy cython3 awscli htop git unzip
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL sklearn pandas plumbum
rsync -r ../outbrain ubuntu@$HOST:~

