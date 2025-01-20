# PTI CELL


## Description

This project implements detectron2 for instance segmentation in the specific case of bacteryocites.  
In collaboration with the author of [AugP-creatis](https://github.com/AugP-creatis) who is focusing on the [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) framework (that is based on [detectron2](https://github.com/facebookresearch/detectron2) framework)


TODO: SECTION IN CONSTRUCTION


## Summary
- [Installation](#installation)
- [New features](#new-features)
- [Usage](#usage)
    - [Notebooks](#notebooks)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Exemples](#exemples)


## Installation
Detailed instructions can be found below to install PTI_Cell. This 
It requires a specific detectron2 commit in order to be as compatible as possible with [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) : commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543)

### Requirements
- Python=3.11
- PyTorch
- Pytorch-cuda=11.8
- Torchvision
- OpenCV, optional, needed by demo and visualization

Please adapt the pytorch-cuda version according to your hardware specifications. Must be compatible with your CUDA install.

### Our versions

| **Library**        | **Version** |
|--------------------|-------------|
| **PyTorch**        | 2.5.1       |
| **CUDA**           | 11.8        |
| **Detectron2**     | 0.1.3       |

### Detailed instructions
NB : Names in capital letters must be changed into your own names depending on your system, paths,...

1. Creation of a conda environment  
Commented commands need to be used if the environment must be in a specific location due to your system management.
```BibTeX
conda create -N ENV_NAME python=3.11        # conda create --prefix /PATH_TO_ENV/ENV_NAME python=3.11
conda activate ENV_NAME                     # conda activate /PATH_TO_ENV/ENV_NAME 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
```

2. Clone and install git repository 
```BibTeX
cd LOCAL_PATH_TO_GIT_CLONE
git clone https://github.com/elsa-cell/PTI_cell.git
```

3. Finish install
```BibTeX
pip install -e PTI_cell
pip install opencv-python natsort jupyterlab        # jupyterlab is optional
```
### Testing installation
To test the installation, run the jupyter notebook at main/simple_inference_resnet_on_one_img.ipynb

### Troubleshooting
If there are some issues with running the code, try the following:

```BibTeX
conda install -c conda-forge libstdcxx-ng
```
libstdcxx-ng is a version of the standard C++ library that might be missing. Be sure to be in the environment you set up before running this command.  
  
Use the proper version of pytorch-cuda and cudatoolkit for your hardware.  
  
If there is a problem with this install, please also refer to [INSTALL.md](INSTALL.md), the official detectron2 documentation.

## New features
- Possible to load a stack of images
- Creation of a new meta-architecture for a stack of images : Z direction taken into account
- Feature separation at the end of the backbone
- Backbone for stack as 3D images

### Newly available networks

The meta-architecture created to deal with stacks is called GeneralizedRCNN_Z.


TODO: mettre image du pipeline de traitement par la nouvelle architecture


| **Name**              | **Main convolution type**  | **Batch norm (BN)**  | **Separator**         | **Config file**   |
|-----------------------|----------------------------|----------------------|-----------------------|-------------------|
| [**mask_rcnn_z_50**](configs/Segmentation-Z/mask_rcnn_z_50.yaml)    | Conv2d                     |  BN2d                | SharedConvSeparator   | configs/Segmentation-Z/mask_rcnn_z_50.yaml |
| [**mask_rcnn_3d**](configs/Segmentation-Z/mask_rcnn_3d.yaml)      | Conv3d                     |  BN3d                | From3dTo2d            | configs/Segmentation-Z/mask_rcnn_3d.yaml |

NB: The number of layers of the resnet of the backbone can be set to different values. As specified in the name of the config, it is by default set to 50.

## Usage

### Notebooks
The notebooks are all inside the directory main in the repository.

#### First segmentation
The notebook simple_inference_resnet_on_one_img.ipynb takes you through the basic segmentation of a particular image using the default mask rcnn, with a 50 layers resnet, available .
We see that it was not trained to detected bacteryocites. 




TODO : SECTION IN CONSTRUCTION











### Training

Exemples of commands to train the model are detailed below.
For more command options, refer to [detectron2 quick start guide](GETTING_STARTED.md) and the help :
```BibTeX
python tools/train_net.py -h
```

#### Single GPU
An exemple of a command used to launch training on a single GPU is shown below. If the system only has one GPU, not need for *CUDA_VISIBLE_DEVICES=X* with X the id of the GPU that will be used. *PATH_TO_OUTPUT_DIRECTORY* can be omitted as it is set to "./output" by default.
```BibTeX
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python tools/train_net.py    \
 --config-file configs/Segmentation-Z/mask_rcnn_3d.yaml    \
 OUTPUT_DIR /PATH_TO_OUTPUT_DIRECTORY    \
 SOLVER.IMS_PER_BATCH 3
```


#### Multi-GPU
An exemple of a command used to launch training on several GPU on the same machine is shown below.  
- *--dist-url* must be set to *"auto"*
- *SOLVER.IMS_PER_BATCH* has to be divisible by *--num-gpus*
- *--num-gpus* should be the number of id specified in *CUDA_VISIBLE_DEVICES*

```BibTeX
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 python tools/train_net.py    \
 --config-file configs/Segmentation-Z/mask_rcnn_3d.yaml    \
 --num-gpus 3    \
 --dist-url "auto"    \
 OUTPUT_DIR /PATH_TO_OUTPUT_DIRECTORY    \
 SOLVER.IMS_PER_BATCH 3
```


#### Tunable parameters
Some configuration settings are very interesting to modify for executing a specific training. This list is not exhaustive but contains the most important settings.

| **Option**                        | **Default value**     | **Comment**   |
|-----------------------------------|-----------------------|---------------|
| **OUTPUT_DIR**                    | ./output              |               |
| **MODEL.WEIGHTS**                 |  ""                   | Full path to weights. Usefull if resume training or eval-only |
| **MODEL.RESNETS.DEPTH**           | 50                    | For lighter models, use 18. Possible values : 18, 32, 50, 101, 152 |
| **SOLVER.MAX_ITER**               | 10000                 | Number of iterations |
| **SOLVER.IMS_PER_BATCH**          | 2                     | Has to be a **multiple of --num_gpus**. Can be increased as long as the model and data fit into the GPU |
| **SOLVER.CHECKPOINT_PERIOD**      | SOLVER.MAX_ITER // 20 | Period when the weights are saved in the output directory |
| **SOLVER.BASE_LR**                | 0.001                 | Base Learning Rate |
| **MODEL.USE_AMP**                 | False                 | [**Not currently supported**](#automatic-mixed-precision) Use Automatic Mixed Precision |
| **MODEL.ROI_HEADS.SCORE_THRESH_TEST** | 0.7               | Detection threshold |
| **TEST.EVAL_PERIOD**              | SOLVER.CHECKPOINT_PERIOD | Run evaluation on valid dataset (in DATASETS.TEST) 
| **TEST.COMPUTE_LOSSES**           | False                 | **Not fully tested** Compute losses on validation dataset every TEST.EVAL_PERIOD iteration.

#### Overcoming training limitations

##### Speeding up process
In order to reduce training time, multi-GPU training is recommended.  
It is even possible to run a multi-machine training. We did not implemented multi-machine training.  

##### CUDA OutOfMemory
To reduce GPU VRAM usage, it is possible to train a lighter model. Consider using MODEL.RESNETS.DEPTH set to 18.  
Another possibility is to reduce the number of input per batch. It is important to remember that in our specific case, one input corresponds in reality to 11 images. 

##### Automatic mixed precision
Automatic mixed precision was successfully implemented in the [AugP-creatis/AdelaiDet-Z repository](https://github.com/AugP-creatis/AdelaiDet-Z) we collaborate with. However, we have an error when adapting it to the detectron2 framework used here. This feature still needs debug.  
  
Automatic mixed precision : automatically casts operations to float16 instead of float32 while maintaining maximal accuracy. It is typically used for model weights and gradients. The losses are rescaled in order not prevent numerical underflow. This method makes the **training faster** and takes **less memory**.



### Evaluation


TODO: SECTION IN CONSTRUCTION


