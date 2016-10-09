sudo locale-gen en_AU.UTF-8
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy cython3 awscli htop git unzip \
    python3-scipy mosh libblas-dev liblapack-dev libatlas-base-dev gfortran

wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
sudo bash Anaconda3-4.2.0-Linux-x86_64.sh
conda create --name outbrain
source activate outbrain

pip install --upgrade plumbum boto boto3 luigi ml-metrics
conda install ipython pandas scipy scikit-learn numpy

git clone https://github.com/ririw/outbrain.git
rsync -r ../outbrain ubuntu@$HOST:~


PYTHONPATH=. luigi --local-scheduler --workers=3 --module outbrain.btb BeatTheBenchmark --test-run