# SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection

## General Information
Sensitivity-Aware Features (SAFE) is a post-hoc addition to a pretrained network with a residual backbone. SAFE identifies and selects a subset of powerful layers for Out-of-Distribution (OOD) object detection, resulting in substantial improvements over prior work.

This repository contains code to replicate the main results of the 2023 IEEE/CVF International Conference on Computer Vision (ICCV) paper:

**[SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Wilson_SAFE_Sensitivity-Aware_Features_for_Out-of-Distribution_Object_Detection_ICCV_2023_paper.html)**

*Samuel Wilson, Tobias Fischer, Feras Dayoub, Dimity Miller, Niko Suenderhauf*

## Acknowledgements
Whilst significant modifications have been made, this repository is heavily based off of the work by [Xuefeng Du](https://github.com/d12306) et al. please be sure to support their work. This repository is a modified version of the [VOS repository](https://github.com/deeplearning-wisc/vos) to allow for sequential evaluation of multiple datasets and enable post-hoc OOD detection. 

## Environment Setup
We heavily recommend using [conda](https://docs.conda.io/en/latest/) for installation. 

We have included the environment.yaml file for installing the conda environment. To create the environment, run:
```bash
conda env create -f environment.yaml
```
Once the installation is complete, you will be able to activate the environment by running:
```bash
conda activate vos_detr
```
Please ensure that the vos_detr environment is active before attempting to run the code contained within this repository.

## Datasets
The location of the datasets is flexible with the use of the *--dataset-dir* argument. For dataset structure, SAFE assumes that all datasets are contained within the directory defined by the *--dataset-dir* argument. Please refer to the steps in the [VOS repository](https://github.com/deeplearning-wisc/vos) in order to acquire the datasets. Upon completion of these steps, your *--dataset-dir* directory should have the following structure:
<br>
    └── DATASET_DIR
        └── VOC_0712_converted
            |
            ├── JPEGImages
            ├── voc0712_train_all.json
            └── val_coco_format.json
        └── COCO
            |
            ├── annotations
                ├── xxx.json (the original json files)
                ├── instances_val2017_ood_wrt_bdd_rm_overlap.json
                └── instances_val2017_ood_rm_overlap.json
            ├── train2017
            └── val2017
        └── bdd100k
            |
            ├── images
            ├── val_bdd_converted.json
            └── train_bdd_converted.json
        └── OpenImages
            |
            ├── coco_classes
            └── ood_classes_rm_overlap


## Base Object Detectors
SAFE does not apply modifications to the base network, neither during training nor inference. Thus, this repository does not support training of new object detectors. Please refer to the [VOS repository](https://github.com/deeplearning-wisc/vos) for instructions on training base object detector networks. 

Pretrained weights for the Faster-RCNN object detectors are available through the [VOS repository](https://github.com/deeplearning-wisc/vos) while pretrained weights for the DETR object detectors are available through the [SIREN repository](https://github.com/deeplearning-wisc/siren). The results from our manuscript report the performance of SAFE using the **vanilla** trained networks. 

In order to use the pretrained models, move the model to the directory with the corresponding model+dataset combination. 

For the RCNN models from the VOS repository:
```bash
mv path/to/weights/file.pth SAFE_ROOT/data/{VOC\|BDD}-Detection/faster-rcnn/{regnetx_}vanilla/model_final.pth
```

For the DETR models from the SIREN repository:
```bash
mv path/to/weights/file.pth SAFE_ROOT/ckpts/detr/checkpoint_{voc\|bdd}_vanilla.pth
```

## Feature Extraction
**Note:** Feature extraction can consume large amounts of disk space if repeated multiple times, particularly on the BDD100K dataset. When extracting features from BDD100K, please ensure there is at least 75GB of available disk space before commencing. For VOC, this number is smaller at only 3GB.

To start, ensure that you have correctly retrieved the pretrained model weights from VOS and/or SIREN and have moved them to the appropriate folders.

To start feature extraction on either of the training datasets, ensure you are in the root directory of the repository. From the root directoy, run SAFE in extract mode:
```bash
python SAFE_interface.py 
--task extract
--variant {RCNN\|DETR}
--bbone {RN50\|RGX4}
--tdset {VOC\|BDD}
--dataset-dir path/to/data/dir
--transform-weight {integer}
```
"variant" selects the object detector architecture. Can be either "RCNN" or "DETR".
"bbone" defines the backbone model architecture. Can be either "RN50" (ResNet50) or "RGX4" (RegNetX4.0). Ignored if "variant" is set to "DETR".
"tdset" defines the target ID dataset. Can be either "VOC" or "BDD".
"transform-weight" modifies the epsilon value for the adversarial perturbation. Defaults to 8. 

Upon completion of the script, a new folder named "safe" will appear in your **--dataset-dir** directory containing the hdf5 files of the extracted features. 

**Note!** To avoid accidental "training-on-test" errors, this repository disallows feature extraction on the OOD or ID testing datasets.

## SAFE Training
Once feature extraction has been completed, a SAFE meta-classifier can be trained by running:
```bash
python SAFE_interface.py 
--task train
--variant {RCNN\|DETR}
--bbone {RN50\|RGX4}
--tdset {VOC\|BDD}
--dataset-dir path/to/data/dir
--transform-weight {1-255}
```
Please ensure that all the arguments passed (excluding "task") in the previous step are the same in this step. Once complete, a SAFE MLP will appear in the "ckpts" directory.

## Evaluation
Finally, to evaluate the newly-trained SAFE MLP run:
```bash
python SAFE_interface.py 
--task train
--variant {RCNN\|DETR}
--bbone {RN50\|RGX4}
--tdset {VOC\|BDD}
--dataset-dir path/to/data/dir
--mlp-path path/to/mlp.pth
```

The evaluation metrics will be displayed in the terminal once all datasets have been processed. 

We provide a set of pretrained [SAFE models]() and [minimalist datasets]() for those just wanting to get started. Once downloaded, extract the SAFE models into the "ckpts" directory and the hdf5 datasets to the "safe" directory within **--dataset-dir**.The evaluation can then be run as normal by following the instructions above.

## Citation ##

If you find this work useful, please consider citing:
```text
@InProceedings{Wilson_2023_ICCV,
    author    = {Wilson, Samuel and Fischer, Tobias and Dayoub, Feras and Miller, Dimity and S\"underhauf, Niko},
    title     = {SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23565-23576}
}
```

