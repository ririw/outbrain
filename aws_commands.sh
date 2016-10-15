export HOST=54.66.6.194
ssh ubuntu@$HOST sudo locale-gen en_AU.UTF-8
ssh ubuntu@$HOST sudo apt-get update
ssh ubuntu@$HOST sudo apt-get install -y python3-pip awscli htop git unzip mosh libblas-dev liblapack-dev libatlas-base-dev gfortran zip

wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
sudo bash Anaconda3-4.2.0-Linux-x86_64.sh
conda create --name outbrain
source activate outbrain

ssh ubuntu@$HOST NPY_NUM_BUILD_JOBS=16 sudo pip3 install numpy scipy cython
ssh ubuntu@$HOST sudo pip3 install ipython pandas scikit-learn jupyter
ssh ubuntu@$HOST sudo pip3 install plumbum boto boto3 luigi ml-metrics tqdm coloredlogs joblib

sudo mkfs.ext4 /dev/xvdb
sudo mount /dev/xvdb /mnt
sudo chown ubuntu /mnt

git clone https://github.com/ririw/outbrain.git
rsync -r ../outbrain ubuntu@$HOST:~


PYTHONPATH=. luigi --local-scheduler --workers=3 --module outbrain.libffm LibFFMRun --test-run
PYTHONPATH=. luigi --local-scheduler --workers=3 --module outbrain.btb BeatTheBenchmark