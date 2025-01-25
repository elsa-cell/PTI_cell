# PTI CELL


## Description

This project implements detectron2 for instance segmentation in the specific case of bacteryocites.  
In collaboration with the author of [AugP-creatis](https://github.com/AugP-creatis) who is focusing on the [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) framework (that is based on [detectron2](https://github.com/facebookresearch/detectron2) framework)


TODO: SECTION IN CONSTRUCTION

TODO : 
- La description générale 
- Préciser les new features et dire contribution augP
- la section organisation du data
- la section des notebooks et les notebooks
- parler de tensorboard pour visualiser


## Summary
- [Installation](#installation)
- [New features](#new-features)
- [Usage](#usage)
    - [Data](#data)
    - [Notebooks](#notebooks)
    - [Training](#training)
    - [Validation](#validation)
    - [Evaluation](#evaluation)



## Installation
[Detailed instructions](#detailed-instructions) can be found below to install PTI_Cell.  
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

### Implemented
- Possible to load a stack of images
- Creation of a new meta-architecture for a stack of images : Z direction taken into account
- Feature separation at the end of the backbone
- Backbone for stack as 3D images

### In progress

- Computing validation losses during training : see [LossEvalHook](detectron2/engine/hooks.py), inspired by [ortegatron](https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b) (from [this](https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e) medium post) and modified as depicted in the comment of the class.
- Adding segmentation metrics to the coco detection metrics : see [CustomEvaluator](detectron2/evaluation/custom_evaluation.py)

### To be implemented
Evaluation by stack needs to be implemented as evaluating slices independently artificially decreases the performances of the network (when detection is performed on an adjacent sloce for exemple).


### Newly available networks

The meta-architecture created to deal with stacks is called GeneralizedRCNN_Z.


TODO: mettre image du pipeline de traitement par la nouvelle architecture


| **Name**              | **Main convolution type**  | **Batch norm (BN)**  | **Separator**         | **Config file**   |
|-----------------------|----------------------------|----------------------|-----------------------|-------------------|
| [**mask_rcnn_z_50**](configs/Segmentation-Z/mask_rcnn_z_50.yaml)    | Conv2d                     |  BN2d                | SharedConvSeparator   | configs/Segmentation-Z/mask_rcnn_z_50.yaml |
| [**mask_rcnn_3d**](configs/Segmentation-Z/mask_rcnn_3d.yaml)      | Conv3d                     |  BN3d                | From3dTo2d            | configs/Segmentation-Z/mask_rcnn_3d.yaml |

NB: The number of layers of the resnet of the backbone can be set to different values. As specified in the name of the config, it is by default set to 50.

## Usage

### Data


TODO : SECTION A FAIRE


### Notebooks
The notebooks are all inside the directory main in the repository.

#### First segmentation
The notebook simple_inference_resnet_on_one_img.ipynb takes you through the basic segmentation of a particular image using the default mask rcnn, with a 50 layers resnet, available .
We see that it was not trained to detect bacteryocites. 




TODO : SECTION IN CONSTRUCTION











### Training

Exemples of commands to train the model are detailed below.
For more command options, refer to [detectron2 quick start guide](GETTING_STARTED.md) and the help :
```BibTeX
python tools/train_net.py -h
```

Please, change the default value of **--data-dir** to the path of your data directory.

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
| **SOLVER.IMS_PER_BATCH**          | 2                     | Has to be a **multiple of --num-gpus**. Can be increased as long as the model and data fit into the GPU |
| **SOLVER.CHECKPOINT_PERIOD**      | SOLVER.MAX_ITER // 20 | Period when the weights are saved in the output directory |
| **SOLVER.BASE_LR**                | 0.001                 | Base Learning Rate |
| **MODEL.USE_AMP**                 | False                 | Use Automatic Mixed Precision  [**(Not currently supported)**](#automatic-mixed-precision) |
| **MODEL.ROI_HEADS.SCORE_THRESH_TEST** | 0.7               | Detection threshold |
| **TEST.EVAL_PERIOD**              | SOLVER.CHECKPOINT_PERIOD | Run evaluation on valid dataset (in DATASETS.TEST) 
| **TEST.COMPUTE_LOSSES**           | False                 | **Not fully tested** Compute losses on validation dataset every TEST.EVAL_PERIOD iteration.

#### Overcoming training limitations

##### Speeding up process
In order to reduce training time, multi-GPU training is recommended.  
It is even possible to run a multi-machine training. We did not implemented multi-machine training.  

##### CUDA OutOfMemory error
To reduce GPU VRAM usage, it is possible to train a lighter model. Consider using MODEL.RESNETS.DEPTH set to 18.  
Another possibility is to reduce the number of input per batch. It is important to remember that in our specific case, one input corresponds in reality to 11 images. 

##### Automatic mixed precision
Automatic mixed precision was successfully implemented in the [AugP-creatis/AdelaiDet-Z repository](https://github.com/AugP-creatis/AdelaiDet-Z) we collaborate with. However, we have an error when adapting it to the detectron2 framework used here. This feature still needs debug.  
  
Automatic mixed precision : automatically casts operations to float16 instead of float32 while maintaining maximal accuracy. It is typically used for model weights and gradients. The losses are rescaled in order not prevent numerical underflow. This method makes the **training faster** and takes **less memory**.



### Validation

It is possible to set *validation* on the detectron2 framework. It is very important to notice that *validation* is called *evaluation* in the detectron2 framework, even if it is not. Moreover, the default name for the validation dataset is in **DATASET.TEST**. We should therefore use this variable but store the name of our validation dataset inside this configuration option.

However, this compute metrics with the evaluator registered by the dataset. It does not compute the loss to see when overfitting happens. We built a specific custom hook called [LossEvalHook](detectron2/engine/hooks.py) in order to do it during training. It is inspired by [ortegatron](https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b) (from [this](https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e) medium post) and modified as depicted in the comment of the class.  
Implementating this hook caused a CUDA OutOfMemory error. Therefore, the hook shall be further optimized, or GPU with bigger memory space shall be used. That is why the option **TEST.COMPUTE_LOSSES** is disabled by default (see [Tunable parameters](#tunable-parameters) in Training section).


To automatically validate the best weights of the model based on a specific criteria, we provide the script [*tools/valid_net.py*](tools/valid_net.py). It is very usefull when no evaluation was set during training with TEST.EVAL_PERIOD at 0, or when a new type of evaluator is needed and was not set during training. Please note that **the index of the cross-validation must be the same**, otherwise, the wrong part of the dataset will be used as the validation dataset and the validation will be biased. 

The validation script loads the config that was used during training and evaluates the model for each weight file present in the specified directory. At the end of the process, it outputs the name of the best weight file, according to the specific criteria. For now, only coco evaluation is possible. We are working on building a new evaluator called [CustomEvaluator](detectron2/evaluation/custom_evaluation.py) that is specific for stacks and includes Dice scores as well.

#### Commands

Here is an exemple of a command used for validation purposes :
```BibTeX
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python tools/valid_net.py   \  
--num-gpus 1   \
--dist-url "auto"   /
--weight-dir-path /tmp/TEST/outputs/3D_50_layers/   \
OUTPUT_DIR /tmp/TEST/outputs/3D_50_layers/   \
SOLVER.IMS_PER_BATCH 1
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.5
```



It stores all the metrics computed by the evaluator in the file *validation_metrics.json*, in the specified **OUTPUT.DIR**

As for the training script, please modify the default value of **--data-dir** to your specific location.

#### Interesting parameters
Some configuration settings are very interesting to modify for executing a specific training. This list is not exhaustive but contains the most important settings.



| **Option**                        | **Default value**     | **Comment**   |
|-----------------------------------|-----------------------|---------------|
| **--weight-dir-path**                    | /tmp/TEST/outputs/3D_50_layers/              |   path to the directory containing the full config file created when training, as well as different versions of the model weights            |
| **--valid-type**                 |  'segm'                   | Can be set to 'segm' or to 'bbox' to have metrics based on the segmentations or the bounding boxes. Only valid for coco evauator |
| **--valid-category**           | 'AP75'                    | Can be set to 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Intact_Sharp', 'AP-Broken_Sharp'. Only valid for coco evauator |
| **OUTPUT_DIR**                    | ./output              |               |
| **SOLVER.IMS_PER_BATCH**          | 2                     | Has to be a **multiple of --num_-pus**. Can be increased as long as the model and data fit into the GPU |
| **MODEL.ROI_HEADS.SCORE_THRESH_TEST** | 0.2               | Detection threshold |
| **TEST.COMPUTE_LOSSES**           | False                 | **Not fully tested** Compute losses on validation dataset every TEST.EVAL_PERIOD iteration.


For more options, refer to the help :
```BibTeX
python tools/valid_net.py -h
```


### Evaluation

To run the evaluation of the model using the weights given by the validation process, adapt the folliwing command.

Here is an exemple of a command used for evaluation purposes :
```BibTeX
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python tools/valid_net.py    \
--num-gpus 1     \
--dist-url "auto"     \
--weight-dir-path /tmp/TEST/outputs/test  \
--weight-file  model_final.pth  \
--eval-only   \
SOLVER.IMS_PER_BATCH 1 \
OUTPUT_DIR /tmp/TEST/outputs/test
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.5
```

Note that as the script used is the same as the validation script, you can use the same [options](#interesting-parameters). 2 parameters are added and must be specified to run the evaluation : **--eval-only** (no value needed), and **--weight-file** with the appropriate name, including the file extension but without the whole path, that is already contained in *--weight-dir-path*.

The evaluated metrics are displayed on the command line and stored in **OUTPUT_DIR/test_metrics.json**

