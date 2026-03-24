### Create a new conda environment
conda create -n pifu-multiview python=3.9 -y
conda activate pifu-multiview

### Install torch, torchvision, and torchaudio with CUDA 11.3 support
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

### Install other dependencies
conda install pillow -y
conda install scikit-image -y
conda install tqdm -y
conda install -c menpo opencv -y
conda install matplotlib -y
conda install numpy -y

### Run the test script
sh ./scripts/test.sh