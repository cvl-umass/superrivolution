# SuperRivolution: Fine-Scale Rivers from Coarse Temporal Satellite Imagery
This repository contains the code for training and evaluating SuperRivolution models. We also provide the corresponding dataset that contains time series Sentinel data matched with high-resolution water segmentation labels.

## Overview
SuperRivolution leverages abundant low-resolution temporal data to improve performance on river segmentation. (a) High-resolution (HR) data are precise but scarce and expensive to acquire. (b) Low-resolution (LR) satellite imagery are freely available and abundant, although of lower quality. (c) A standard model using a single LR image produces predictions with significant errors (top row, errors in red). (d) By fusing information from multiple LR images, our proposed approach generates significantly more accurate segmentations and more reliable river width estimates, reducing prediction errors and closing the gap with HR models. 

![results](assets/superriv_overview.png)

## Environment Setup
1. Create a conda environment: `conda create -n superriv python=3.9`
2. Activate environment: `conda activate superriv`
3. Install all packages in the env with: `pip install -r requirements.txt`

## Run training on SuperRivolution
1. Download necessary checkpoints: [link](https://drive.google.com/drive/folders/1NNz4Qg2Ao62GUe_NAllv2BjSHYVJegf_?usp=drive_link). 
    - Place it in checkpoints such that it looks something like: superrivolution/checkpoints/moco_v3/*.pth
2. Download the [SuperRivolution data](https://drive.google.com/file/d/1yzyXPdzCVI6exUdEMJFS0Su_7ccX6zXc/view?usp=sharing)
3. Run training: `python 01_train.py --dist-url 'tcp://127.0.0.1:8001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --sr_type <sr-type> --data_dir <data path from prev step>`
    - `sr_type` can be one of: `model` (super-resolution),`input` (input upsampling),`output` (output upsampling)
    - You can additionally specify the segment_model, backbone, head, resize_size (resize_size depends on which model to use).  
    - To specify other parameters according to the table you can use the following arguments. We include options in the table below (each parameter corresponds to the first four columns in the table).
        - `--segment_model`
        - `--backbone`
        - `--head`
        - `--resize_size`
    - By default, training will use linear adapter, but this can be changed by specifying `--adaptor` to one of the following: `linear`, `no_init`, `drop`
    - The training will output 3 files: 1 log file, the latest checkpoint file, and the best checkpoint file (based on validation F1 score)
    - Results will be saved to `results/s2-water` by default (you can change it in `--out` parameter)
    - The file `01_script_train_s2_*.sh` can also be run as a batch job. `*` depends on the type of method to use (options: `input` for input upsampling, `output` for output upsampling, `sr_1im` for super-resolution using 1 image, `sr_8im` for super-resolution using 8 images). This script also contains the optimal learning rates for each of the model configurations.
<!-- 
If you do not wish to train your own model, you can also use pre-trained model checkpoints below. The corresponding optimal threshold based on the validation set is also included.
| segment_model| backbone               | head          | resize_size| checkpoint   | thresh | Description | 
| ---          | ---                    | ---           | ---        | ---           | ---           |---           |
| deeplabv3    | mobilenet_v2           | no_head       | 512 | [link]() | 0.3 | ImageNet1k pre-trained |
| deeplabv3    | resnet50               | no_head       | 512 | [link]() | 0.4 | ImageNet1k pre-trained |
| deeplabv3    | resnet50_mocov3        | no_head       | 512 | [link]() | 0.5 | MocoV3 pre-trained |
| deeplabv3    | resnet50_seco          | no_head       | 512 | [link]() | 0.2 | SeCo pre-trained |
| deeplabv3    | swinb                  | no_head       | 224 | [link]() | 0.5 | ImageNet1k pre-trained |
| deeplabv3    | swint                  | no_head       | 224 | [link]() | 0.5 | ImageNet1k pre-trained |
| dpt          | vitb                   | no_head       | 224 | [link]() | 0.2 | ImageNet1k pre-trained |
| dpt          | vitb_clip              | no_head       | 224 | [link]() | 0.3 | CLIP pre-trained  |
| dpt          | vitb_dino              | no_head       | 224 | [link]() | 0.4 | DINO pre-trained |
| dpt          | vitb_mocov3            | no_head       | 224 | [link]() | 0.1 | MocoV3 pre-trained |
| dpt          | vitl                   | no_head       | 224 | [link]() | 0.2 | ImageNet1k pre-trained |
| fpn          | mobilenet_v2           | no_head       | 512 | [link]() | 0.2 | ImageNet1k pre-trained |
| fpn          | resnet50               | no_head       | 512 | [link]() | 0.3 | ImageNet1k pre-trained |
| fpn          | resnet50_mocov3        | no_head       | 512 | [link]() | 0.3 | MocoV3 pre-trained |
| fpn          | resnet50_seco          | no_head       | 512 | [link]() | 0.3 | SeCo pre-trained |
| fpn          | satlas_si_resnet50     | satlas_head   | 512 | [link]() | 0.4 | SatlasPretrain pre-trained |
| fpn          | satlas_si_swinb        | satlas_head   | 512 | [link]() | 0.4 | SatlasPretrain pre-trained |
| fpn          | satlas_si_swint        | satlas_head   | 512 | [link]() | 0.3 | SatlasPretrain pre-trained |
| fpn          | swinb                  | no_head       | 224 | [link]() | 0.4 | ImageNet1k pre-trained |
| fpn          | swint                  | no_head       | 224 | [link]() | 0.4 | ImageNet1k pre-trained |
| unet         | mobilenet_v2           | no_head       | 512 | [link]() | 0.3 | ImageNet1k pre-trained |
| unet         | resnet50               | no_head       | 512 | [link]() | 0.2 | ImageNet1k pre-trained |
| unet         | resnet50_mocov3        | no_head       | 512 | [link]() | 0.4 | MocoV3 pre-trained |
| unet         | resnet50_seco          | no_head       | 512 | [link]() | 0.3 | SeCo pre-trained |
| unet         | swinb                  | no_head       | 224 | [link]() | 0.3 | ImageNet1k pre-trained |
| unet         | swint                  | no_head       | 224 | [link]() | 0.3 | ImageNet1k pre-trained | -->

## Run evaluation on SuperRivolution
### Water Segmentation
There are two options:
1. Run: `python 02_water_seg_eval.py --to_save_imgs 1 --ckpt_path <ckpt path> --to_ensemble <number of images to ensemble>`
    - This script will automatically find the best threshold using the RiverScope validation set
    - Alternatively, you can specify a threshold, e.g., `python 02_water_seg_eval.py --ckpt_path <ckpt path> --thresh 0.5`
    - This will save the metrics to the --out (default is results/s2-seg) with filename the same as the checkpoint name
    - Setting `--to_save_imgs 1` saves both the csv and the segmentation mask per image in the folder `results/s2-seg/<ckpt_name>`. Saving of the segmentation masks is needed for calculating the river width estimates.


### River Width Estimation
1. Run: `python 03_river_width_estimate.py --data_dir <data path of dataset> --raster_src "results/s2-seg-<method>/<ckpt_name>--<num_ensemble>" --is_gt 0 --raster_idx -1`
    - This will save results to specified --out (by default, it's results/predicted-widths). Inside the folder with the same name as the checkpoint
    - This will produce two sets of files per image: 1 csv file containing estimated width per node ("width_m" column is the estimated width in meters for the corresponding "node_id"), 1 png file to visualize estimated widths
2. Ground truth widths are available at 'SuperRivolution_dataset/PlanetScope/derived_gt_widths-test-wdate.csv' for a given node_id, reach_id, and date


## Baselines
1. We refer to the [RiverScope repository](https://github.com/cvl-umass/riverscope-models) for running the training and evaluation of the RiverScope and Sentinel baselines

## Citation
If you found this useful, please consider citing our work:
```
@inproceedings{daroya2026superrivolution,
  author={Daroya, Rangel and Maji, Subhransu},
  title={SuperRivolution: Fine-Scale Rivers from Coarse Temporal Satellite Imagery},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2026}
}
```
