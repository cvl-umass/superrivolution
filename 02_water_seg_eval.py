import os
import torch
import rasterio
import cv2
import numpy as np
import torch.nn as nn
from models.get_model import get_model
import numpy as np
import pandas as pd
from utils_dir import mkdir_p 

from dataset.sentinel2_sr import Sentinel2SR
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from loguru import logger
from datetime import datetime
from tqdm import tqdm
import argparse
from skimage.transform import rescale, resize

EPS=1e-7




parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--data_dir', default='SuperRivolution_dataset', type=str, help='Path to dataset')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('--is_distrib', default=True, type=int, help='Batch size')
parser.add_argument('--to_save_imgs', default=1, type=int, help='Set to 1 to save image outputs')
parser.add_argument('--is_downsample', default=0, type=int, help='Set to 1 to downsample then upsample input images (to simulate lower res)')
parser.add_argument('--thresh', default=None, type=float, help='Set to None to find threshold from val set. Otherwise, set to optimal thresh value 0-1')
parser.add_argument('--tasks', default=["water_mask"], nargs='+', help='Task(s) to be trained')
parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint')

parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/s2-seg', help='Directory to output the result')
parser.add_argument('--to_ensemble', default=0, type=int, help='Set to 2-8 to use average of multiple images given by number. 0 otherwise.')



def save_imgs(sr_type, out_img_dir, input_fps, test_labels, pred_water_mask, to_save_rgb=True, to_save_gt=True):
    test_labels_np = test_labels.detach().cpu().numpy()
    pred_mask_np = pred_water_mask.detach().cpu().numpy()
    for idx in range(len(input_fps)):
        input_fp = input_fps[idx]
        test_label = test_labels_np[idx]    # 0 and 1 values
        pred_mask = pred_mask_np[idx]       # 0 and 1 values
        out_name = input_fp.split("/")[-4:]
        out_name = "--".join(out_name)
        out_fp_tif = os.path.join(out_img_dir, out_name)
        # logger.debug(f"out_fp_tif: {out_fp_tif}")

        input_dataset = rasterio.open(input_fp)
        # Write prediction to TIFF
        kwargs = input_dataset.meta
        kwargs.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')
        # logger.debug(f"kwargs: {kwargs}")
        with rasterio.open(out_fp_tif, 'w', **kwargs) as dst:
            dst.write_band(1, pred_mask.astype(rasterio.float32))
        # Write prediction to PNG
        out_fp_png = out_fp_tif.replace(".tif", ".png")
        cv2.imwrite(out_fp_png, pred_mask*255)


        # Write GT to TIFF
        if to_save_gt:
            out_fp_gt_png = out_fp_png.replace(".png", "--gt.png")
            cv2.imwrite(out_fp_gt_png, test_label*255)
            out_fp_gt_tif = out_fp_gt_png.replace(".png", ".tif")
            # logger.debug(f"out_fp_gt_tif: {out_fp_gt_tif}")
            with rasterio.open(out_fp_gt_tif, 'w', **kwargs) as dst:
                dst.write_band(1, test_label.astype(rasterio.float32))

        # RGB image
        if to_save_rgb:
            if sr_type in ["model", "model_1im"]:
                # img = cv2.imread(input_fp)
                img = input_dataset.read()[(3,2,1),:,:]
                img = np.transpose(img, (1,2,0))
            else:
                input_dataset = rasterio.open(input_fp)
                img = input_dataset.read()[(3,2,1),:,:]
                img = np.transpose(img, (1,2,0))
                # logger.debug(f"img: {img.shape}")
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            out_name_rgb = out_fp_tif.replace(".tif", "--rgb.png")
            cv2.imwrite(out_name_rgb, img[:,:,:3]*255)
    

opt = parser.parse_args()
logger.debug(f"opt: {opt}")

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)

to_save_imgs = (opt.to_save_imgs!=0)
num_ensemble = opt.to_ensemble if opt.to_ensemble>1 else 1
logger.debug(f"num_ensemble: {num_ensemble}")
# out_dir = "/".join(opt.ckpt_path.split("/")[:-1])
out_dir = opt.out
out_img_dir = os.path.join(out_dir, opt.ckpt_path.split("/")[-1].split(".")[0]+f"--{num_ensemble}")
logger.debug(f"to_save_imgs: {to_save_imgs}")
if to_save_imgs:
    logger.debug(f"Creating output folder for images: {out_img_dir}")
    mkdir_p(out_img_dir)
if num_ensemble>1:
    logger.warning(f"Setting batch size to 1 since ensembling method")
    opt.batch_size = 1
tasks = opt.tasks
# num_inp_feats = ckpt_opt.num_channels   # number of channels in input
tasks_outputs_tmp = {
    "water_mask": 1,
}
tasks_outputs = {t: tasks_outputs_tmp[t] for t in tasks}
logger.debug(f"opt: {opt.__dict__}")


logger.debug(f"Loading weights from {opt.ckpt_path}")
checkpoint = torch.load(opt.ckpt_path, weights_only=False)
ckpt_fn = opt.ckpt_path.split("/")[-1].replace(".pth.tar", "")
logger.debug(f"ckpt_fn: {ckpt_fn}")
ckpt_opt = checkpoint["opt"]
ckpt_lr = ckpt_opt.lr
logger.debug(f"ckpt_opt: {ckpt_opt}")
logger.debug(f"ckpt_lr: {ckpt_lr}")
model = get_model(ckpt_opt, tasks_outputs=tasks_outputs, num_inp_feats=ckpt_opt.num_channels, pretrained=(ckpt_opt.pretrained==1))


sr_type = ckpt_opt.sr_type if "sr_type" in ckpt_opt.__dict__ else "model"
logger.debug(f"sr_type: {sr_type}")

# if opt.is_distrib:
new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
checkpoint["state_dict"] = new_ckpt
tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)

logger.debug(f"After loading ckpt: {tmp}")
logger.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")
model.cuda()
model.eval()

# Find optimal threshold using validation set
# """
if opt.thresh is None:
    val_dataset1 = Sentinel2SR(backbone=ckpt_opt.backbone, num_ensemble=num_ensemble, segment_model=ckpt_opt.segment_model, sr_type=sr_type, root=opt.data_dir, split="valid", resize_size=ckpt_opt.resize_size, adaptor=ckpt_opt.adaptor, num_channels=ckpt_opt.num_channels)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset1, batch_size=opt.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=val_sampler, drop_last=False)
    val_batch = len(val_loader)
    val_dataset = iter(val_loader)
    logger.debug(f"Evaluating on {val_batch} val batches to find best threshold")
    rgbs = []
    thresh_choices = np.arange(0,1,0.1)
    thresh_metrics = {t: {th: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for th in thresh_choices} for t in tasks}
    counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels
    with torch.no_grad():
        for thresh in thresh_choices:
            print(f"Evaluating on thresh={thresh}")
            val_dataset = iter(val_loader)
            for k in tqdm(range(val_batch)):
                val_data, val_labels = next(val_dataset)
                val_labels["water_mask"] = val_labels["water_mask"].cuda()
                target = torch.squeeze(val_labels["water_mask"], 1)
                
                if num_ensemble == 1:
                    val_data = val_data.cuda()
                    val_pred, feat = model(val_data, feat=True)
                    water_pred = val_pred["water_mask"]
                else:
                    all_preds = []
                    # logger.debug(f"val_data: {val_data.shape}")
                    for i in range(val_data.shape[1]):
                        val_data_samp = val_data[:,i,:,:,:]
                        val_data_samp = val_data_samp.cuda()
                        val_pred, feat = model(val_data_samp, feat=True)
                        pred = val_pred["water_mask"]
                        all_preds.append(pred)
                    # TODO: explore other ensembling options?
                    water_pred = torch.mean(torch.stack(all_preds, axis=0), axis=0)
                    # logger.debug(f"water_pred: {water_pred.shape}")
                pred = torch.squeeze(water_pred, 1)
                thresh_pred = torch.where(pred > thresh, 1., 0.)

                TP = torch.sum(torch.round(torch.clip(target * thresh_pred, 0, 1)))
                FP = torch.sum(torch.round(torch.clip((1-target) * thresh_pred, 0, 1))) # target is 0, but pred is 1
                FN = torch.sum(torch.round(torch.clip(target * (1-thresh_pred), 0, 1))) # target is 1, but pred is 0
                TN = torch.sum(torch.round(torch.clip((1-target) * (1-thresh_pred), 0, 1))) # target is 0, and pred is 0
                
                thresh_metrics["water_mask"][thresh]["TP"].append(TP.item())
                thresh_metrics["water_mask"][thresh]["FN"].append(FN.item())
                thresh_metrics["water_mask"][thresh]["FP"].append(FP.item())
                thresh_metrics["water_mask"][thresh]["TN"].append(TN.item())
                s1,s2,s3 = pred.shape
                num_px = s1*s2*s3
                assert num_px == (TP+FN+FP+TN)
                thresh_metrics["water_mask"][thresh]["num_px"].append(num_px)

    metric_names = ["f1", "rec", "prec", "acc", "iou"]
    task_metric_per_thresh = {t: {m: [] for m in metric_names} for t in tasks}
    for t in tasks:
        for th in thresh_choices:
            TP_tot = np.sum(np.array(thresh_metrics[t][th]["TP"]))
            FP_tot = np.sum(np.array(thresh_metrics[t][th]["FP"]))
            FN_tot = np.sum(np.array(thresh_metrics[t][th]["FN"]))
            TN_tot = np.sum(np.array(thresh_metrics[t][th]["TN"]))
            prec = TP_tot/(TP_tot + FP_tot + EPS)
            rec = TP_tot/(TP_tot + FN_tot + EPS)
            f1 = (2*prec*rec)/(prec+rec + EPS)
            acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
            miou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
            task_metric_per_thresh[t]["f1"].append(f1)
            task_metric_per_thresh[t]["rec"].append(rec)
            task_metric_per_thresh[t]["prec"].append(prec)
            task_metric_per_thresh[t]["acc"].append(acc)
            task_metric_per_thresh[t]["iou"].append(miou)
    # thresh_metrics
    # task_metric_per_thresh

    # Find optimal threshold given metrics (based on f1 score)
    optim_threshes = {t:None for t in tasks}
    for t in tasks:
        optim_idx = np.argmax(task_metric_per_thresh[t]["f1"])
        optim_thresh = thresh_choices[optim_idx]
        optim_threshes[t] = optim_thresh
        print(f"{t} optim thresh: {optim_thresh} [f1: {task_metric_per_thresh[t]['f1'][optim_idx]}]")
    logger.debug(f"optim_threshes: {optim_threshes}")
# """
else:
    optim_threshes = {"water_mask": opt.thresh}
logger.debug(f"Using the following thresholds: {optim_threshes}")

test_dataset1 = Sentinel2SR(
    backbone=ckpt_opt.backbone,
    num_ensemble=num_ensemble, segment_model=ckpt_opt.segment_model, sr_type=sr_type, 
    root=opt.data_dir, split="test", resize_size=ckpt_opt.resize_size, adaptor=ckpt_opt.adaptor, 
    return_fp=True, num_channels=ckpt_opt.num_channels, return_clouds=True,
)
logger.debug(f"Using batch size 1 for test loader")
test_sampler = None
test_loader = torch.utils.data.DataLoader(
    test_dataset1, batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True, sampler=test_sampler, drop_last=False)
test_batch = len(test_loader)
test_dataset = iter(test_loader)


logger.debug(f"Evaluating on {test_batch} test batches")
metrics = {t: {"f1":[], "rec":[], "prec":[], "acc": []} for t in tasks}
counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels

fmask_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}
mndwi_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}
model_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}

gtpreds = {t: {"gt":[], "pred":[]} for t in tasks}


rgbs = []
per_img_results = []
with torch.no_grad():
    for k in tqdm(range(test_batch)):
        test_data, test_labels, test_clouds, input_fp, input_fp_ref = next(test_dataset)
        # logger.debug(f"input_fp: {input_fp}")
        # logger.debug(f"input_fp_ref: {input_fp_ref}")
        # logger.debug(f"test_clouds: {test_clouds.shape} {torch.unique(test_clouds)} {torch.sum(test_clouds)}")
        has_cloud = True if torch.sum(test_clouds)>0 else False
        has_cloud10 = True if torch.sum(test_clouds)>0.1*512*512*test_clouds.shape[0] else False
        has_cloud30 = True if torch.sum(test_clouds)>0.3*512*512*test_clouds.shape[0] else False
        has_cloud50 = True if torch.sum(test_clouds)>0.5*512*512*test_clouds.shape[0] else False
        # TODO: classify predictions into with and without clouds

        # logger.debug(f"input_fp: {input_fp}")
        if sr_type in ["input", "output"]:
            tile_fp = input_fp[0].replace("Sentinel-2-timeseries/reprojected/", "PlanetScope/input/").replace("--tile", "-tile")
            tile_fp_parts = tile_fp.split("/")
            tile_fp_parts[-1] = tile_fp_parts[-1].split("--")[-1]
            tile_fp = "/".join(tile_fp_parts)
            # logger.debug(f"tile_fp: {tile_fp}. {os.path.exists(tile_fp)}")
            planet_tile_data_firstchann = rasterio.open(tile_fp).read()[0][None, :,:]
            planet_nonzero_mask = (planet_tile_data_firstchann!=0).astype(int)  # NOTE: to use for metrics
        else:
            if sr_type == "model":
                tile_fp = input_fp[0].replace("Sentinel-2-SR/", "PlanetScope/input/").replace("--tile", "-tile").replace("/sr.png", "")
            elif sr_type == "model_1im":
                tile_fp = input_fp[0].replace("Sentinel-2-SR-1im/", "PlanetScope/input/").replace("--tile", "-tile").replace("/sr.png", "")
            tile_fp_parts = tile_fp.split("/")
            tile_fp_parts[-1] = tile_fp_parts[-1].split("--")[-1]
            tile_fp = "/".join(tile_fp_parts)
            assert os.path.exists(tile_fp), tile_fp
            # logger.debug(f"tile_fp: {tile_fp}. {os.path.exists(tile_fp)}")
            planet_tile_data_firstchann = rasterio.open(tile_fp).read()[0][None, :,:]
            planet_nonzero_mask = (planet_tile_data_firstchann!=0).astype(int)  # NOTE: to use for metrics
        
        
        gt_water_mask = test_labels["water_mask"] 
        gt_water_mask = torch.squeeze(gt_water_mask, 1).cuda()
        if num_ensemble == 1:
            test_data = test_data.cuda()
            test_pred, feat = model(test_data, feat=True)

            pred = test_pred["water_mask"]
            thresh_pred = torch.where(pred > optim_threshes["water_mask"], 1., 0.)
            pred_water_mask = thresh_pred
            pred_water_mask = torch.squeeze(pred_water_mask, 1)
            assert (len(planet_nonzero_mask.shape) == len(pred_water_mask.shape)), f"planet_nonzero_mask: {planet_nonzero_mask.shape}, pred_water_mask: {pred_water_mask.shape}"
            planet_nonzero_mask = resize(planet_nonzero_mask, pred_water_mask.shape, order=0, preserve_range=True, mode='reflect', anti_aliasing=False)
            planet_nonzero_mask = torch.from_numpy(planet_nonzero_mask).cuda()
            pred_water_mask = pred_water_mask*planet_nonzero_mask
        else:
            # logger.debug(f"test_data.shape: {test_data.shape}")
            all_preds = []
            for i in range(test_data.shape[1]):
                test_data_samp = test_data[:,i,:,:,:]
                test_data_samp = test_data_samp.cuda()
                test_pred, feat = model(test_data_samp, feat=True)
                pred = test_pred["water_mask"]
                all_preds.append(pred)
            ensem_pred = torch.mean(torch.stack(all_preds, axis=0), axis=0)
            thresh_pred = torch.where(ensem_pred > optim_threshes["water_mask"], 1., 0.)
            pred_water_mask_samp = thresh_pred
            pred_water_mask = torch.squeeze(pred_water_mask_samp, 1)

        model_TP = torch.sum(torch.round(torch.clip(gt_water_mask * pred_water_mask, 0, 1)))
        model_FP = torch.sum(torch.round(torch.clip((1-gt_water_mask) * pred_water_mask, 0, 1))) # gt_water_mask is 0, but pred is 1
        model_FN = torch.sum(torch.round(torch.clip(gt_water_mask * (1-pred_water_mask), 0, 1))) # gt_water_mask is 1, but pred is 0
        model_TN = torch.sum(torch.round(torch.clip((1-gt_water_mask) * (1-pred_water_mask), 0, 1))) # gt_water_mask is 0, and pred is 0
        model_counts_tps["TP"].append(model_TP.item())
        model_counts_tps["FP"].append(model_FP.item())
        model_counts_tps["FN"].append(model_FN.item())
        model_counts_tps["TN"].append(model_TN.item())

        img_prec = model_TP/(model_TP + model_FP + EPS)
        img_rec = model_TP/(model_TP + model_FN + EPS)
        img_f1 = (2*img_prec*img_rec)/(img_prec+img_rec + EPS)
        img_iou = model_TP/(model_TP + model_FP + model_FN + EPS)
        out_name = input_fp[0].split("/")[-3:]
        out_name = "--".join(out_name)
        perc_water = torch.sum(test_labels["water_mask"])/(500*500)
        per_img_results.append([input_fp[0], input_fp_ref[0], out_name, has_cloud, has_cloud10, has_cloud30, has_cloud50, model_TP.item(), model_FP.item(), model_FN.item(), model_TN.item(), img_f1.item(), img_prec.item(), img_rec.item(), img_iou.item(), perc_water.item()])

        if to_save_imgs:
            save_imgs(sr_type, out_img_dir, input_fp_ref, test_labels["water_mask"], pred_water_mask)

# NOTE: Uncomment the next 3 lines to save the metric per image
out_fp_perimg = os.path.join(out_dir, f"PERIMG_{ckpt_fn}--{num_ensemble}.csv")
per_img_results_df = pd.DataFrame(per_img_results, columns=["input_fp", "input_fp_ref", "out_name", "has_cloud", "has_cloud10", "has_cloud30", "has_cloud50", "TP", "FP", "FN", "TN", "f1", "prec", "rec", "iou", "perc_water"])
per_img_results_df.to_csv(out_fp_perimg, index=False)

print(f"model,optim_thresh_water,ckpt_lr,f1,rec,prec,acc,miou")

optim_thresh_water = optim_threshes["water_mask"]

TP_tot = np.sum(np.array(model_counts_tps["TP"]))
FP_tot = np.sum(np.array(model_counts_tps["FP"]))
FN_tot = np.sum(np.array(model_counts_tps["FN"]))
TN_tot = np.sum(np.array(model_counts_tps["TN"]))
prec = TP_tot/(TP_tot + FP_tot + EPS)
rec = TP_tot/(TP_tot + FN_tot + EPS)
f1 = (2*prec*rec)/(prec+rec + EPS)
acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
miou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
print(f"model,{optim_thresh_water},{ckpt_lr},{f1},{rec},{prec},{acc},{miou}")

out_fp = os.path.join(out_dir, f"{ckpt_fn}--{num_ensemble}.csv")
logger.debug(f"Saving results to {out_fp}")
out_df_data = [[
    ckpt_opt.segment_model, ckpt_opt.backbone, ckpt_opt.head, ckpt_opt.adaptor, ckpt_opt.resize_size, 
    optim_thresh_water, ckpt_lr, f1, rec, prec, acc, miou, opt.ckpt_path
]]
out_df = pd.DataFrame(out_df_data, columns=[
    "segment_model", "backbone", "head", "adaptor", "resize_size",
    "thresh", "lr", "f1", "rec", "prec", "acc", "miou", "ckpt_path"
])
out_df.to_csv(out_fp, index=False)
