git clone https://github.com/celsofranssa/AttentionXML.git
cd AttentionXML/
git checkout updated

# configure python env
export DEBIAN_FRONTEND=noninteractive
sudo apt update -y
sudo apt install python3-pip -y
sudo apt install python3-virtualenv -y
virtualenv venv -p $(which python3)

source venv/bin/activate
pip install -r requirements.txt
python3 -m nltk.downloader punkt

# data
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# cuda
sudo apt update -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install nvtop





