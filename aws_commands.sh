export HOST=54.253.102.189
ssh ubuntu@$HOST sudo locale-gen en_AU.UTF-8
ssh ubuntu@$HOST sudo apt-get update
ssh ubuntu@$HOST sudo apt-get install -y python3-pip awscli htop git unzip mosh libblas-dev liblapack-dev libatla=s-base-dev gfortran zip vowpal-wabbit

ssh ubuntu@$HOST wget https://raw.githubusercontent.com/Cretezy/Swap/master/swap.sh
ssh ubuntu@$HOST NPY_NUM_BUILD_JOBS=16 sudo pip3 install numpy scipy cython
ssh ubuntu@$HOST sudo pip3 install ipython pandas scikit-learn jupyter
ssh ubuntu@$HOST sudo pip3 install plumbum boto boto3 luigi ml-metrics tqdm coloredlogs joblib

sudo mkfs.ext4 /dev/xvdb
sudo mount /dev/xvdb /mnt
sudo chown ubuntu /mnt

rsync -r ../outbrain ubuntu@$HOST:~

cd outbrain
sudo python3 setup.py install

PYTHONPATH=. luigi --local-scheduler --workers=3 --module outbrain.datasets ClicksDataset
PYTHONPATH=. luigi --local-scheduler --module outbrain.libffm LibFFMClassifier --test-run
PYTHONPATH=. luigi --local-scheduler --module outbrain.libffm VWClassifier --test-run
PYTHONPATH=. luigi --local-scheduler --module outbrain.btb BeatTheBenchmark --test-run
