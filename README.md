# Temporally Consistent Enhancement of Low-light Videos via Spatial-Temporal Compatible Learning

## Abstract
Temporal inconsistency is the annoying artifact that has been commonly introduced in low-light video enhancement, but current methods tend to overlook the significance of utilizing both data-centric clues and model-centric design to tackle this problem. In this context, our work makes a comprehensive exploration from the following three aspects. First, to enrich the scene diversity and motion flexibility, we construct a synthetic diverse low/normal-light paired video dataset with a carefully designed low-light simulation strategy, which can effectively complement existing real captured datasets. Second, for better temporal dependency utilization, we develop a Temporally Consistent Enhancer Network (TCE-Net) that consists of stacked 3D convolutions and 2D convolutions to exploit spatial-temporal clues in videos. Last, the temporal dynamic feature dependencies are exploited to obtain consistency constraints for different frame indexes. All these efforts are powered by a Spatial-Temporal Compatible Learning (STCL) optimization technique, which dynamically constructs specific training loss functions adaptively on different datasets. As such, multiple-frame information can be effectively utilized and different levels of information from the network can be feasibly integrated, thus expanding the synergies on different kinds of data and offering visually better results in terms of illumination distribution, color consistency, texture details, and temporal coherence.

<img src="./Figure/Proposed_framework.pdf" width="900"/>

## Dataset

### SDSD dataset
We use the original SDSD datasets with dynamic scenes.

And you can download the SDSD-indoor and SDSD-outdoor from [link](https://jiaya.me/publication/).

### SMID dataset
We use its full images for SMID and transfer the RAWdata to RGB since our work explores low-light image enhancement in the RGB domain.

You can download our processed datasets from [link](https://jiaya.me/publication/).

### DS-LOL dataset
You can download our synthetic low-light video dataset from [link](https://jiaya.me/publication/).

## Project Setup

First install Python 3. We advise you to install Python 3 and PyTorch with Anaconda:

```
conda create --name py36 python=3.6
source activate py36
```

Clone the repo and install the complementary requirements:
```
cd $HOME
pip install -r requirements.txt
```

## Usage
### Train
Train the model on the corresponding dataset using the train config. For example, the training on indoor subset of SDSD:

### Test
We use PSNR and SSIM as the metrics for evaluation. Evaluate the model on the corresponding dataset using the test config.

### Pre-trained Model
You can download our trained model using the following links:

the model trained with indoor subset in SDSD: indoor_G.pth

the model trained with outdoor subset in SDSD: outdoor_G.pth

the model trained with SMID: smid_G.pth


## Citation Information
If you find the project useful, please cite:

## Acknowledgments


