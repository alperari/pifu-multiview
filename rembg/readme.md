### Setup env
conda create -n py311 python=3.11 -y
conda activate py311

pip install "rembg[cpu]" # for library
pip install "rembg[cpu,cli]" # for library + cli

### Remove background from a local file:
rembg i path/to/input.png path/to/output.png

e.g.
rembg i \
    -m birefnet-massive \
    input_images/steve2.jpg \
    output_images/steve2_no_bg_birefnet-massive.jpg

### Return only the mask:
rembg i -om path/to/input.png path/to/output.png

e.g.
rembg i -om \
    output_images/steve1_no_bg.jpg \
    output_images/steve1_no_bg_mask.png


### Specify the model to use:
rembg i -m u2net_human_seg  path/to/input.png path/to/output.png

### Batch run over a folder (via python script):
python run.py

# Available Models
* u2net: A pre-trained model for general use cases.
* u2netp: A lightweight version of u2net model.
* u2net_human_seg: A pre-trained model for human segmentation.
* u2net_cloth_seg: A pre-trained model for Cloths Parsing from human portrait. Here clothes are parsed into 3 category: Upper body, Lower body and Full body.
* silueta: Same as u2net but the size is reduced to 43Mb.
* isnet-general-use: A new pre-trained model for general use cases.
* isnet-anime: A high-accuracy segmentation for anime character.
* sam (download encoder, download decoder, source): A pre-trained model for any use cases.
* birefnet-general: A pre-trained model for general use cases.
* birefnet-general-lite: A light pre-trained model for general use cases.
* birefnet-portrait: A pre-trained model for human portraits.
* birefnet-dis: A pre-trained model for dichotomous image segmentation (DIS).
* birefnet-hrsod: A pre-trained model for high-resolution salient object detection (HRSOD).
* birefnet-cod: A pre-trained model for concealed object detection (COD).
* birefnet-massive: A pre-trained model with massive dataset.
* bria-rmbg: A state-of-the-art background removal model by BRIA AI.