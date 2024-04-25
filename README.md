
  

# COMP3340 Group 10 - Pre-training

  

  

## Contact

  

- This repository contains code for various pre-training methods on COMP3340 course project CNN flower classification

  

- For any question and enquiry, please feel free to reach out to HU Zhongyu (hzyalex@connect.hku.hk)

  

- Thanks and enjoy =P

  

  

## Overview

  

**Prerequisite for Reproduction**

  

1. [Set up conda environment](#env_setup)

  

2. [Download data and checkpoint files and put them under the correct folder](#downloads)

  

3. [Run the commands to reproduce all the important results](#cmd_repro)

  

  

**Software, Hardware & System Requirements**

  

- Software

  

- Set up environment as [following](#env_setup)

  

- python==3.8.18

  

- mmfewshot==0.1.0

  

- mmdet==2.17.0

  

- mmcv==1.3.14

  

- Hardware

  

- Experiments are conducted on one NVIDIA GeForce RTX 2080 Ti

  

- System

  

- Linux

  

  

**Note**

  

One model training typically takes 6-7 hours to run with one NVIDIA GeForce RTX 2080 Ti.

  

  

## Environment setup <a id="env_setup"/>

  

  

### Basic Setup (Also required by some other Group 10 repos)

  

  

**Step 1. Create virtual environment using anaconda**

  

  

```

  

conda create -n open-mmlab python=3.8 -y

  

conda activate open-mmlab

  

```

  

  

*Please make sure that you are create a virtual env with python version 3.8*

  

  

**Step 2 Install Pytorch from wheel**

  

  

```

  

wget https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-linux_x86_64.whl#sha256=709cec07bb34735bcf49ad1d631e4d90d29fa56fe23ac9768089c854367a1ac9

  

pip install torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl

  

```

  

  

*Please double check that you install the correct version of pytorch using the following command*

  

  

![Output if correct pytorch version is installed](./check_torch.png)

  

  

**Step 3 Install cudatoolkit via conda-forge channel**

  

  

*You must be on the GPU compute node to install cudatoolkit and mmcv since GCC compiler and CUDA drivers only available on GPU computing nodes*

  

  

```

  

gpu-interactive

  

conda activate open-mmlab

  

conda install -c conda-forge cudatoolkit=11.0

  

```

  

  

**Step 4 Install torchvision, mmcv-full and mmcls package using pip**

  

  

*Make sure you are on GPU compute node!!*

  

  

-  `gpu-interactive`

  

  

*Make sure you did not previously installed any relevant package*

  

*Following pip show command show output a message like "no such package found"*

  

  

```

  

pip show torchvision

  

pip show mmcv

  

pip show mmcv-full

  

pip show mmcls

  

```

  

  

*remove pip cache*

  

  

```

  

pip cache remove torchvision

  

pip cache remove mmcv

  

pip cache remove mmcv-full

  

pip cache remove mmcls

  

```

  

  

*install packages*

  

  

```

  

pip install torchvision==0.8.2

  

pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

  

```

  

install mmclassification (mmcls-0.x)

  

git clone https://github.com/open-mmlab/mmclassification.git

cd mmclassification

pip install -v -e .

  

check installation

  

python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu

  

  

  
  

  

## Download data & checkpoints

  

  

[OneDrive Download Link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/hzyalex_connect_hku_hk/EpNZZ3rCklFKudY2cWxwn54BGoyFiUSC2EKE7E2zAX851Q?e=gCwQ5Q)

Contains:

- Oxford-17 Flower Dataset

- Oxford-102 Flower Dataset

- Various saved trained model (as .pth checkpoint files) and training logs

- Open-source pretrained model checkpoints

  

## Move data to right place

1. Download and unzip the "data" folder

2. Move everything inside the "data" folder under mmclassification/data

  

## Move checkpoints to right place

1. Download and unzip the "output_pretrain" folder

2. Move everything inside into "mmclassification/output"

3. Download and unzip the "pretrain_checkpoint" folder

4. Move everything inside into "mmclassification/resources" folder

  

## Train & test

how to train

  

python tools/train.py --config [CONFIG FILE PATH] --work-dir [OUTPUT DIRECTORY PATH]

  

how to test

  

python tools/test.py --config [CONFIG FILE PATH] --checkpoint [.pth CHECKPOINT FILE PATH] --out [OUTPUT TEXT FILE PATH]

  

## How to reproduce & evaluate our experiments

If you want to reproduce our experiment, just plug **config file path** and **output directory path** into the train command above

  

If you want to test trained model, just plug **config file path** and **output directory path + "/latest.pth"** into the test command above.

  

Also, corresponding **training log files** are inside the output folders

1. Baseline: ResNet18, train from scratch for 200 epochs on oxford flower 17 dataset

  

config file： mmclassification/configs/resnet/resnet18_flowers_bs128.py

  

output directory：mmclassification/output/resnet18_flowers_bs128

  

2. Finetune ResNet18 (pretrained on imagenet1k) on oxford flower 17

config:

mmclassification/configs/resnet/resnet18_flowers_finetune.py

output:

mmclassification/output/resnet18_flower_pretrain_modified_parameter

  

3. Finetune ResNet18 (pretrained on cifar10) on oxford flower 17

config: mmclassification/configs/resnet/resnet18_flowers_finetune_cifar10.py

output:

mmclassification/output/resnet18_pretrain_cifar10

  

4. ResNet50 trained from scratch on oxford flower 17

config:

mmclassification/configs/resnet/resnet50_flowers.py

output:

mmclassification/output/resnet50

5. Finetune ResNet50 (pretrained on cifar10) on oxford flower 17

  

config:

mmclassification/configs/resnet/resnet50_flowers_finetune_cifar10.py

output:

mmclassification/output/resnet50_flower_cifar10

6. Finetune ResNet50 (pretrained on cifar100) on oxford flower 17

config:

mmclassification/configs/resnet/resnet50_flowers_finetune_cifar100.py

output:

mmclassification/output/resnet50_flower_cifar100

  

7. Finetune ResNet50 (pretrained on imagenet1k) on oxford flower 17

config:

mmclassification/configs/resnet/resnet50_flowers_imagenet1k.py

output:

mmclassification/output/resnet50_flower_imagenet1k

  

8. Finetune ResNet50 (pretrained on imagenet21k) on oxford flower 17

config:

mmclassification/configs/resnet/resnet50_flowers_imagenet21k.py

output:

mmclassification/output/resnet50_flower_imagenet21k

  

9. ResNet152 trained from scratch on oxford flower 17

config:

mmclassification/configs/resnet/resnet152_flowers.py

output:

mmclassification/output/resnet152

10. Finetune ResNet152 (pretrained on cifar10) on oxford flower 17

config:

mmclassification/configs/resnet/resnet152_flowers_finetune_cifar10.py

output:

mmclassification/output/resnet152_flower_cifar10