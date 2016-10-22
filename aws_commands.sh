export HOST=54.253.102.189
ssh ubuntu@$HOST sudo locale-gen en_AU.UTF-8
ssh ubuntu@$HOST sudo apt-get update
ssh ubuntu@$HOST sudo apt-get install -y python3-pip awscli htop git unzip mosh libblas-dev liblapack-dev libatlas-base-dev gfortran zip

ssh ubuntu@$HOST wget https://raw.githubusercontent.com/Cretezy/Swap/master/swap.sh
ssh ubuntu@$HOST NPY_NUM_BUILD_JOBS=16 sudo pip3 install numpy scipy cython
ssh ubuntu@$HOST sudo pip3 install ipython pandas scikit-learn jupyter
ssh ubuntu@$HOST sudo pip3 install plumbum boto boto3 luigi ml-metrics tqdm coloredlogs joblib

sudo apt-get install -y libboost-program-options-dev libboost-python-dev zlib1g-dev build-essential libtool automake vowpal-wabbit
git clone git://github.com/JohnLangford/vowpal_wabbit.git
cd vowpal_wabbit
#./autogen.sh
#./configure
make -j 8
sudo make install

sudo mkfs.ext4 /dev/xvdb
sudo mount /dev/xvdb /mnt
sudo chown ubuntu /mnt

rsync -r ../outbrain ubuntu@$HOST:~

cd outbrain
sudo python3 setup.py install

PYTHONPATH=. luigi --local-scheduler --workers=1 --module outbrain.datasets ClicksDataset
PYTHONPATH=. luigi --local-scheduler --module outbrain.libffm LibFFMClassifier --test-run
PYTHONPATH=. luigi --local-scheduler --module outbrain.libffm VWClassifier --test-run
PYTHONPATH=. luigi --local-scheduler --module outbrain.deepbrain KerasClassifier --test-run --small
PYTHONPATH=. luigi --local-scheduler --module outbrain.btb BeatTheBenchmark --test-run

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL
