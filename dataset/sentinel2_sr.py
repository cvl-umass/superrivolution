from torch.utils.data.dataset import Dataset

import rasterio
from glob import glob
import os
import torch
import fnmatch
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import torch.nn.functional as F
from loguru import logger
from skimage.transform import resize

import h5py
import json
import pickle
import ast


def closest_divisible_by_d(n, d=32): # for FPN models that require multiples  of 32
    # Calculate the closest integer divisible by 32
    quotient = n // d
    lower = quotient * d
    upper = (quotient + 1) * d

    # Choose the closest one
    if abs(n - lower) <= abs(n - upper):
        return lower
    else:
        return upper

class Sentinel2SR(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="data_dir/",  # path for data splits
        split='train', 
        adaptor="linear",
        num_channels=4,
        sr_type="model",
        return_fp = False,
        resize_size=None,
        segment_model="unet",
        num_ensemble=1,
        return_clouds=False,
        backbone="resnet50",
    ):
        assert split in ["train", "valid", "test"]
        # NOTE for label: 0=background, 1=river, 2=other water
        self.num_outputs = 1    # river vs not river
        self.label_col_name = "label_path"  #png

        self.num_channels = num_channels   # RGB+NIR
        self.split = split
        self.return_fp = return_fp
        self.sr_type = sr_type
        self.root = root
        self.segment_model = segment_model
        self.return_clouds = return_clouds
        
        all_fns_fp = os.path.join(root, f"{split}.csv")
        all_fns = pd.read_csv(all_fns_fp)
        self.fns = all_fns

        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]

        self.adaptor = adaptor
        self.num_ensemble = num_ensemble
        self.final_size = final_size
        self.backbone = backbone
        logger.debug(f"Using adaptor: {self.adaptor}")

        # if self.sr_type in ["model", "model_1im"]:
        #     self.num_channels = 3
        #     logger.warning(f"Changing num_channels to 3 since using SR model")
        
    def __getitem__(self, index):
        data_row = self.fns.iloc[index]
        label_fp = os.path.join(self.root, data_row[self.label_col_name])

        label = rasterio.open(label_fp).read()
        label = np.transpose(label, (1,2,0))    # (500,500,1)
        # label = np.where(label==1,1,0)  # NOTE: only detecting RIVER WATER
        label = np.where(label!=0,1,0)  # NOTE: detecting ALL water

        orig_input_fp_tmp = label_fp.replace("PlanetScope/label", "Sentinel-2-timeseries/reprojected").replace("-tile", "--tile")  # this will return multiple  files since time series
        # NOTE: the images are already bilinearly interpolated
        orig_input_fp_pattern = orig_input_fp_tmp.split("/")
        orig_input_fp_pattern[-1] = "*"+orig_input_fp_pattern[-1]
        orig_input_fps = sorted(glob("/".join(orig_input_fp_pattern)))

        # Select which images to use based on good/bad images
        s2_chunks_tmp = []
        for orig_input_fp in orig_input_fps:
            image = rasterio.open(orig_input_fp).read()
            image = np.transpose(image, (1,2,0))    # (500,500,12)
            image = image[:,:,(1,2,3,7,10,11)]  # R,G,B,NIR,SWIR1,SWIR2 (500,500,6)
            s2_chunks_tmp.append([orig_input_fp, image])
        goods, bads = [], []
        for i,(_,ts) in enumerate(s2_chunks_tmp):
            if [0]*6 in ts:
                bads.append(i)
            else:
                goods.append(i)
        if len(goods) >= self.num_ensemble:
            rand_indices = goods[:self.num_ensemble]
        else:
            need = self.num_ensemble - len(goods)
            rand_indices = goods + bads[:need]
        s2_chunks = [s2_chunks_tmp[i][1] for i in rand_indices]
        s2_chunks_fps = [s2_chunks_tmp[i][0] for i in rand_indices]
        input_fp_ref = s2_chunks_fps[0]
        
        clouds = np.zeros_like(s2_chunks[0])
        if self.sr_type in ["model", "model_1im"]:
            if self.sr_type == "model":
                input_fp_tmp = label_fp.replace("PlanetScope/label", "Sentinel-2-SR").replace("-tile", "--tile").replace(".tif", ".tif/sr.png")
            elif self.sr_type == "model_1im":
                input_fp_tmp = label_fp.replace("PlanetScope/label", "Sentinel-2-SR-1im").replace("-tile", "--tile").replace(".tif", ".tif/sr.png")
            input_fp_pattern = input_fp_tmp.split("/")
            input_fp_pattern[-2] = "*"+input_fp_pattern[-2]
            input_fps = sorted(glob("/".join(input_fp_pattern)))
            assert len(input_fps)==1        # this is for superres image
            input_fp = input_fps[0]

            image = cv2.imread(input_fp)
            image = image[:,:,::-1]     # (500,500,3). BGR to RGB
            
            if self.num_ensemble == 1:  # only use cloud images when num images is 1
                fp_for_clouds = s2_chunks_fps[0]
                im_data = rasterio.open(fp_for_clouds).read()
                image_tmp = np.transpose(im_data, (1,2,0))    # (500,500,14)
                cloud_data_raw = image_tmp[:,:,-2][:,:,None]
                cloud_data = ((cloud_data_raw==9)|((cloud_data_raw==8))).astype(int)
                clouds = resize(cloud_data, (label.shape[0],label.shape[1], 1), order=0, preserve_range=True, mode='reflect', anti_aliasing=False)

            # Reshape input to be same as label (can be diff size due to cropping in label-space)
            assert (len(image.shape) == 3) and image.shape[-1]==3, f"image.shape: {image.shape}"
            image = resize(image, (label.shape[0],label.shape[1], 3), order=0, preserve_range=True, mode='reflect', anti_aliasing=False)
        elif self.sr_type in ["input", "output"]:
            if self.num_ensemble == 1:
                input_fp = s2_chunks_fps[0]
                im_data = rasterio.open(input_fp).read()
                image_tmp = np.transpose(im_data, (1,2,0))    # (500,500,14)
                image = image_tmp[:,:,(1,2,3,7,10,11)]  # R,G,B,NIR,SWIR1,SWIR2 (500,500,6)

                cloud_data_raw = image_tmp[:,:,-2][:,:,None]
                # cloud_high_prob = (im_data[-2]==9).astype(int) # see page 13 of https://step.esa.int/thirdparties/sen2cor/2.5.5/docs/S2-PDGS-MPC-L2A-SUM-V2.5.5_V2.pdf
                # cloud_med_prob = (im_data[-2]==8).astype(int)
                clouds = ((cloud_data_raw==9)|((cloud_data_raw==8))).astype(int)
            else:   # if ensemble of multiple images, apply data transform per image in the ensemble, and don't consider cloud data
                # NOTE: this only happens in evaluation, not in training since training is 1 image input->1 image output (num_ensemble=1 always)
                input_fp = s2_chunks_fps[0]
                data_transforms = transforms.Compose(self.transforms_list)
                # logger.debug(f"data_transforms: {data_transforms}")
                all_images = []
                for image_tmp in s2_chunks:
                    data = np.concatenate((image_tmp,label), axis=-1)
                    trans_data = data_transforms(data)  # output of transforms: (5, 512, 512)
                    
                    image = trans_data[:-1, :, :].float()
                    label_tmp = trans_data[-1, :, :].float()

                    if (torch.max(image)-torch.min(image)):
                        image = image - torch.min(image)
                        image = image / torch.maximum(torch.max(image),torch.tensor(1))
                    else:
                        # logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {input_fp}")
                        image = torch.zeros_like(image).float()
                    all_images.append(image)
                image = np.stack(all_images, axis=0)
                label = label_tmp
                # logger.debug(f"all_images: {all_images.shape}")

        else:
            raise NotImplementedError
        
        data_transforms = transforms.Compose(self.transforms_list)
        if self.num_ensemble == 1:
            data = np.concatenate((image,label,clouds), axis=-1)
            trans_data = data_transforms(data)  # output of transforms: (5, 512, 512)
            
            image = trans_data[:-2, :, :].float()
            label = trans_data[-2, :, :].float()
            clouds = trans_data[-1, :, :].float()

        # logger.debug(f"1image: {image.shape}")
        if self.sr_type == "output":
            # NOTE: model automatically upsamples output so no need to explicitly add bilinear upsampling elsewhere, just need to downsample the input to the original resolution of 10m/px
            if self.num_ensemble == 1:
                # Update to be 10m/px (instead of 3m/px -- upsampled in reprojection)
                image = image.numpy()
                clouds = clouds.numpy()
                # logger.debug(f"image: {image.shape}")
                out_size1 = image.shape[1]*(500/self.final_size)*3//10  #NOTE: 500 is the 3m/px, so resizing to self.final_size would adjust the resolution/px
                out_size2 = image.shape[2]*(500/self.final_size)*3//10  #NOTE: 500 is the 3m/px, so resizing to self.final_size would adjust the resolution/px
                if self.segment_model=="fpn":   # NOTE: required by FPN  to be divisible by 32
                    out_size1 = closest_divisible_by_d(out_size1, d=32)
                    out_size2 = closest_divisible_by_d(out_size2, d=32)
                elif self.segment_model=="dpt":   # NOTE: required by DPT  to be divisible by 16
                    out_size1 = closest_divisible_by_d(out_size1, d=16)
                    out_size2 = closest_divisible_by_d(out_size2, d=16)
                elif self.segment_model=="deeplabv3":   # NOTE: required by deeplabv3  to be divisible by 8
                    out_size1 = closest_divisible_by_d(out_size1, d=8)
                    out_size2 = closest_divisible_by_d(out_size2, d=8)
                assert (len(image.shape)==3) and (image.shape[0]<20), f"image.shape: {image.shape}"
                image = resize(
                    image, 
                    (image.shape[0], out_size1, out_size2), 
                    order=0, 
                    preserve_range=True, 
                    mode='reflect', 
                    anti_aliasing=False
                )
                if ((self.backbone in ["swint", "swinb", "vitb_prithvi"])
                        or (self.segment_model=="dpt")): # For SwinT and SwinB, required to fix to 224 so need to upsample
                    image = resize(
                        image, 
                        (image.shape[0], self.final_size, self.final_size), 
                        order=0, 
                        preserve_range=True, 
                        mode='reflect', 
                        anti_aliasing=False
                    )
                # logger.debug(f"image: {image.shape}")
                image = torch.from_numpy(image)
                if len(clouds.shape)==2:
                    clouds = clouds[None]
                # logger.debug(f"clouds.shape: {clouds.shape}")
                assert len(clouds.shape)==3, f"clouds.shape: {clouds.shape}"
                clouds = resize(
                    clouds, 
                    (clouds.shape[0], out_size1, out_size2), 
                    order=0, 
                    preserve_range=True, 
                    mode='reflect', 
                    anti_aliasing=False
                )
                clouds = torch.from_numpy(clouds)
            else:
                all_images_resized = []
                for image in all_images:
                    # Update to be 10m/px (instead of 3m/px -- upsampled in reprojection)
                    image = image.numpy()
                    # logger.debug(f"image: {image.shape}")
                    out_size1 = image.shape[1]*(500/self.final_size)*3//10  #NOTE: 500 is the 3m/px, so resizing to self.final_size would adjust the resolution/px
                    out_size2 = image.shape[2]*(500/self.final_size)*3//10  #NOTE: 500 is the 3m/px, so resizing to self.final_size would adjust the resolution/px
                    if self.segment_model=="fpn":   # NOTE: required by FPN  to be divisible by 32
                        out_size1 = closest_divisible_by_d(out_size1, d=32)
                        out_size2 = closest_divisible_by_d(out_size2, d=32)
                    elif self.segment_model=="dpt":   # NOTE: required by DPT  to be divisible by 16
                        out_size1 = closest_divisible_by_d(out_size1, d=16)
                        out_size2 = closest_divisible_by_d(out_size2, d=16)
                    elif self.segment_model=="deeplabv3":   # NOTE: required by deeplabv3  to be divisible by 8
                        out_size1 = closest_divisible_by_d(out_size1, d=8)
                        out_size2 = closest_divisible_by_d(out_size2, d=8)
                    assert (len(image.shape)==3) and (image.shape[0]<20), f"image.shape: {image.shape}"
                    image = resize(
                        image, 
                        (image.shape[0], out_size1, out_size2), 
                        order=0, 
                        preserve_range=True, 
                        mode='reflect', 
                        anti_aliasing=False
                    )
                    if ((self.backbone in ["swint", "swinb", "vitb_prithvi"])
                        or (self.segment_model=="dpt")): # For SwinT and SwinB, required to fix to 224 so need to upsample
                        image = resize(
                            image, 
                            (image.shape[0], self.final_size, self.final_size), 
                            order=0, 
                            preserve_range=True, 
                            mode='reflect', 
                            anti_aliasing=False
                        )
                    image = torch.from_numpy(image)
                    all_images_resized.append(image)
                image = np.stack(all_images_resized, axis=0)


        if self.split == "train":
            # Add Random channel mixing
            ccm = torch.eye(self.num_channels)[None,None,:,:]
            r = torch.rand(3,)*0.25 + torch.Tensor([0,1,0])
            filter = r[None, None, :, None]
            ccm = torch.nn.functional.conv2d(ccm, filter, stride=1, padding="same")
            ccm = torch.squeeze(ccm)
            try:
                image = torch.tensordot(ccm, image, dims=([1],[0])) # not exactly the same perhaps
            except:
                logger.warning("Error introducing random channel mixing")
                pass    # NOTE: Error for multispectral images
            
            # Add Gaussian noise
            r = torch.rand(1,1)*0.04
            image = image + torch.normal(mean=0.0, std=r[0][0], size=(image.shape[0],image.shape[1],image.shape[2]))
        
        # Min-max Normalization
        # Normalize data
        # """
        if self.num_ensemble == 1:
            if (torch.max(image)-torch.min(image)):
                image = image - torch.min(image)
                image = image / torch.maximum(torch.max(image),torch.tensor(1))
            else:
                # logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {input_fp}")
                image = torch.zeros_like(image).float()
                label = torch.zeros_like(label).float()
                
            if self.adaptor=="drop":
                image = image[:3, :, :]         # only keep first 3 channels
                image = image[[2,1,0], :, :]    # RGB from BGR
        labels = {
            "water_mask": label.float(),
        }

        if self.return_fp:
            if self.return_clouds:
                return (
                    image,    # shape: (3, 512, 512)
                    labels,
                    clouds,
                    input_fp,
                    input_fp_ref,   # NOTE: this is for cross-referencing results, the path of the first good image
                )

            else:
                return (
                    image,    # shape: (3, 512, 512)
                    labels,
                    input_fp,
                    input_fp_ref,   # NOTE: this is for cross-referencing results, the path of the first good image
                )

        else:
            if self.return_clouds:
                return (
                    image,    # shape: (3, 512, 512)
                    labels,
                    clouds,
                )
            else:
                return (
                    image,    # shape: (3, 512, 512)
                    labels,
                )


    def __len__(self):
        return self.data_len

