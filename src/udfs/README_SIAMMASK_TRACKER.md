
## 1. CLONE SIAMMASK IN EVA DIRECTORY

cd eva
git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
bash make.sh


## 2. RUN THE FOLLOWING COMMANDS TO MAKE SIAMMASK PYTHON MODULES IMPORTABLE:

cd ~/eva
export PYTHONPATH=$PWD:$PYTHONPATH
cd SiamMask
export PYTHONPATH=$PWD:$PYTHONPATH
cd experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
cd ~/eva

---------------------------------------------------------------------------------

## 3. INSTALL THE FOLLOWING PYTHON MODULES IN THE CONDA ENVIRONMENT:

Cython==0.29.4
colorama==0.3.9
requests==2.21.0
fire==0.1.3
matplotlib==2.2.3
numba==0.39.0
scipy==1.1.0
h5py==2.8.0
tqdm==4.29.1

opencv_python==3.4.3.18
torch==0.4.1
torchvision==0.2.1

## 4. DOWNLOAD THE PRETRAINED MODEL

cd SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth

--------------------------------------------------------------------------------------

## 5. ORIGINAL SIAMMASK REPO

https://github.com/foolwood/SiamMask
