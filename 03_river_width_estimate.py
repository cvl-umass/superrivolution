import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import line
# from skimage.measure import regionprops, label
from scipy.ndimage import label
from scipy import ndimage
from tqdm import tqdm
import glob
import geopandas as gpd
import os
import rasterio
import rasterio.features
import rasterio.mask
from shapely.geometry import LineString, Point, box
# from pyproj import Transformer
from rasterio.plot import show

from rasterio.features import shapes
from shapely.geometry import shape
from skimage.transform import resize
import copy

import argparse
from loguru import logger


parser = argparse.ArgumentParser(description='Compute widths for given tif water masks')
parser.add_argument('--data_dir', default='SuperRivolution_dataset', type=str, help='Path to dataset')
parser.add_argument('--out_dir', default="./results/predicted-widths", type=str, help='Folder to save outputs')
parser.add_argument('--raster_src', default=None, type=str, help='Folder of output files to process')
parser.add_argument('--raster_idx', default=0, type=int, help='index to process')
parser.add_argument('--start_raster_idx', default=None, type=int, help='start index to process (inclusive)')
parser.add_argument('--end_raster_idx', default=None, type=int, help='end index to process (inclusive)')
parser.add_argument('--is_gt', default=0, type=int, help='Set to 1 if processing GT water masks. Else set to 0')

# NOTE: Algorithm implemented below is based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8752013


PLANET_RESOLUTION = 3  # means 3m per pixel
# S2_RESOLUTION = 10  # means 10m per pixel
S2_RESOLUTION = 3  # 10m per pixel (NOTE: changed to 3 since upsampled to match to 3m/px)
NUM_NEIGHBORS = 10  # for computing the slope
MIN_PIXELS_ISLAND = 500



def remove_small_islands(binary_mask, min_size=500):
    labeled_mask, num_features = label(binary_mask)
    
    for i in range(1, num_features + 1):
        if np.sum(labeled_mask == i) < min_size:
            binary_mask[labeled_mask == i] = 0
    
    return binary_mask



### Get regions that intersect with SWORD reach
def get_sword_mask_intersection(raster_path, sword_gdf):
    # Load the binary mask raster
    with rasterio.open(raster_path) as src:
        binary_mask = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        extent = src.bounds


    # Ensure the line shapefile has the same CRS as the raster
    if sword_gdf.crs != raster_crs:
        sword_gdf = sword_gdf.to_crs(raster_crs)

    # Identify contiguous regions in the binary mask
    mask_labels, num_labels = ndimage.label(binary_mask)

    # Convert labeled mask regions to polygons
    shapes_list = list(shapes(mask_labels, transform=transform))  # Store generator results

    region_polygons = [shape(geom) for geom, value in shapes_list if value > 0]
    region_ids = [value for _, value in shapes_list if value > 0]  # Extract region IDs correctly

    # Create a GeoDataFrame of regions
    gdf_regions = gpd.GeoDataFrame({'region_id': region_ids, 'geometry': region_polygons}, crs=raster_crs)

    # Find intersecting regions
    intersecting_regions = gdf_regions[gdf_regions.intersects(sword_gdf.unary_union)]
    intersecting_region_ids = intersecting_regions['region_id'].tolist()
    intersecting_mask = np.isin(mask_labels, intersecting_region_ids)
    
    return binary_mask, intersecting_mask

# Function to calculate slope at a point along the line using N points
def get_local_slope(line, point, delta=1, N=10):
    distances = [line.project(point) + delta * (i - N // 2) for i in range(N)]
    coords = [line.interpolate(d).coords[0] for d in distances]
    
    x_vals, y_vals = zip(*coords)
    dx = np.gradient(x_vals)
    dy = np.gradient(y_vals)
    avg_dx = np.mean(dx)
    avg_dy = np.mean(dy)
    slope = np.arctan2(avg_dy, avg_dx)
    return slope  # in radians

# Function to create a perpendicular line at point
def get_perpendicular_segment(point, slope, length=50):
    dx = np.cos(slope + np.pi / 2) * length / 2
    dy = np.sin(slope + np.pi / 2) * length / 2
    p1 = Point(point.x - dx, point.y - dy)
    p2 = Point(point.x + dx, point.y + dy)
    return LineString([p1, p2])



args = parser.parse_args()
logger.debug(f"args: {args}")

if not os.path.exists(args.out_dir):
    logger.debug(f"Creating output directory: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
actual_out_dir = os.path.join(args.out_dir, "--".join(args.raster_src.split("/")[1:]))

if not os.path.exists(actual_out_dir):
    logger.debug(f"Creating output directory: {actual_out_dir}")
    os.makedirs(actual_out_dir, exist_ok=True)

resolution = PLANET_RESOLUTION


planet_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
nodes_fp = os.path.join(args.data_dir, "SWORD", "SWORD_nodes.csv")
nodes_df = pd.read_csv(nodes_fp)
nodes_df["reach_id"] = nodes_df["reach_id"].astype(int)

raster_paths = []
for tmp in glob.glob(os.path.join(args.raster_src, "*.tif")):
    if tmp.endswith("--gt.tif"):
        continue
    raster_paths.append(tmp)
raster_paths = sorted(raster_paths)
logger.info(f"Found {len(raster_paths)} rasters in the output folder")

if args.raster_idx < 0:
    if (args.start_raster_idx is not None) and (args.end_raster_idx is not None):
        raster_idxs = list(range(args.start_raster_idx, args.end_raster_idx+1))    # loop over all rasters
    else:
        raster_idxs = list(range(len(raster_paths)))    # loop over all rasters
else:
    raster_idxs = [args.raster_idx]

logger.debug(f"raster_idxs: {raster_idxs}")
for raster_idx in raster_idxs:
    logger.debug(f"Processing raster_idx: {raster_idx}")
    raster_path = raster_paths[raster_idx]
    raster_name = os.path.basename(raster_path)


    reach_id = int(raster_name.split("--")[4])

    normalized_planetscope_path = os.path.join(
        "PlanetScope/input/test",
        raster_name.replace("--tile","-tile").replace("--","/").split("/")[-1]
    )
    assert os.path.exists(os.path.join(args.data_dir, normalized_planetscope_path)), f"normalized_planetscope_path: {os.path.join(args.data_dir, normalized_planetscope_path)}"
    sword_shapefile_name = planet_df[planet_df["normalized_planetscope_path"]==normalized_planetscope_path]["sword_path"].iloc[0]
    sword_shapefile_path = os.path.join(args.data_dir, sword_shapefile_name)

    logger.debug(f"raster_path: {raster_path}")
    logger.debug(f"reach_id: {reach_id}")
    logger.debug(f"sword_shapefile_path: {sword_shapefile_path}")

    tile_name = "-".join(os.path.basename(raster_path).split("--")[-2:])
    tile_fp = os.path.join(args.data_dir, "PlanetScope/input/test", tile_name)
    assert os.path.exists(tile_fp)


    logger.debug(f"Only keeping water masks intersecting with SWORD reach")
    # Load the shapefile line
    line_gdf = gpd.read_file(sword_shapefile_path)
    binary_mask, intersecting_mask = get_sword_mask_intersection(raster_path, line_gdf)
    orig_mask = binary_mask[:]

    logger.debug(f"Removing small islands with num pixels <={MIN_PIXELS_ISLAND}")
    cleaned_mask = remove_small_islands(intersecting_mask, min_size=MIN_PIXELS_ISLAND)

    # Load points CSV and convert to GeoDataFrame
    filtered_nodes_df = nodes_df[nodes_df["reach_id"]==reach_id][["x", "y", "node_id"]]
    nodes_gdf = gpd.GeoDataFrame(
        filtered_nodes_df,
        geometry=gpd.points_from_xy(filtered_nodes_df.x, filtered_nodes_df.y),
        crs="EPSG:4326"  # update if needed
    )

    # Reproject nodes and sword to raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_extent = box(*raster_bounds)
        raster_aspect = (raster_bounds.right - raster_bounds.left) / (raster_bounds.top - raster_bounds.bottom)
        if raster_crs != line_gdf.crs:
            # Reproject points to raster CRS for later pixel counting
            nodes_gdf_raster = nodes_gdf.to_crs(raster_crs)
            line_gdf_raster = line_gdf.to_crs(raster_crs)

    # Get single merged line geometry (assumes single feature)
    line = line_gdf_raster.geometry.unary_union

    ### Find perpendicular for each node along SWORD reach, and count number of water pixels
    results = []
    perpendicular_lines = []

    raster_data = cleaned_mask.astype(np.uint8)
    raster_src = rasterio.open(raster_path)

    logger.debug(f"Computing perpendicular to node wrt SWORD and estimating width...")
    step = 0.5
    delta = 1
    N = 10

    for _, pt_row in tqdm(nodes_gdf_raster.iterrows()):
        node_id = pt_row["node_id"]
        lon, lat = pt_row["x"], pt_row["y"]
        pt = pt_row["geometry"]
        slope = get_local_slope(line, pt, delta=delta, N=N)
        perp_angle = slope + np.pi / 2  # radians (+90 degrees to get perpendicular)
        dx = np.cos(perp_angle)
        dy = np.sin(perp_angle)

        pt_x, pt_y = pt.x, pt.y
        
        max_dist = 200*step   # max distance to search for along the perpendicular
        search_points = []
        for direction in [-1, 1]:
            for i in range(1, int(max_dist / step)):
                new_x = pt_x + direction * dx * step * i
                new_y = pt_y + direction * dy * step * i
                search_points.append((new_x, new_y))

        closest_water = None
        min_dist = float('inf')
        for (x, y) in search_points:
            try:
                row, col = raster_src.index(x, y)
                if raster_data[row, col] == 1:
                    dist = np.hypot(x - pt_x, y - pt_y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_water = (x, y)
            except IndexError:
                continue
        if closest_water is None:
            continue

        line_coords = []
        all_row_col = []
        row_pt, col_pt = raster_src.index(pt_x, pt_y)
        cw_x, cw_y = closest_water
        if closest_water and (0<=row_pt<raster_data.shape[0] and 0<=col_pt<raster_data.shape[1]): # point should be within bounds of raster    
            counts = {"left": 0, "right": 0}
            if (raster_data[row_pt, col_pt] == 1):  # Check if point is inside water/river
                directions = [(-1, 'left'), (1, 'right')]
                start_x, start_y = pt_x, pt_y   # start with point since it's on water
            else:
                direction = np.sign((cw_x - pt_x) * dx + (cw_y - pt_y) * dy)    # if -1, means
                directions = [(direction, 'left' if direction==-1 else 'right')]
                start_x, start_y = cw_x, cw_y   # start from closest water found
                if direction == 0:
                    water_pixel_count = -1
                    directions = []

            for sign, count_dir in directions:
                i = 1
                while True:
                    new_x = start_x + sign * dx * step * i
                    new_y = start_y + sign * dy * step * i
                    line_coords.append((new_x, new_y))
                    try:
                        row, col = raster_src.index(new_x, new_y)
                        if raster_data[row, col] == 1:
                            all_row_col.append((row,col))
                            counts[count_dir] += 1
                        else:
                            break
                    except IndexError:
                        break
                    i += 1
            water_pixel_count = len(set(all_row_col))
        else:
            water_pixel_count = 0

        results.append({
            "geometry": pt,
            "slope_rad": slope,
            "node_id": node_id,
            "reach_id": reach_id,
            "lon": lon,
            "lat": lat,
            "pt_x": pt_x,
            "pt_y": pt_y,
            "num_water_px": water_pixel_count,
            "width_m": water_pixel_count*resolution,
            "closest_water_x": closest_water[0] if closest_water else None,
            "closest_water_y": closest_water[1] if closest_water else None,
            "closest_water_dist": min_dist if closest_water else None,
            "tile_fp": tile_fp,
        })

        if len(line_coords)>0:
            perpendicular_lines.append(LineString([(pt_x, pt_y)] + line_coords))

    # Create final DataFrame
    if len(results) == 0:
        logger.warning(f"No widths found. raster_idx: {raster_idx}. raster_name: {raster_name}")
        continue
    results_df = pd.DataFrame(results)
    nodes_gdf = gpd.GeoDataFrame(results_df, geometry="geometry", crs=line_gdf_raster.crs)
    if len(perpendicular_lines) > 0:
        lines_gdf = gpd.GeoDataFrame(geometry=perpendicular_lines, crs=line_gdf_raster.crs)

    if args.is_gt==0:
        out_path = os.path.join(actual_out_dir, f"{reach_id}_{raster_idx:03d}_{raster_name.replace('.tif','')}_node_widths.csv")
    else:
        out_path = os.path.join(actual_out_dir, f"{reach_id}_{raster_idx:03d}_{raster_name.replace('.tif','')}_node_widths_gt.csv")
    logger.debug(f"Saving width estimate results to {out_path}")
    results_df.to_csv(out_path, index=False)

    if len(perpendicular_lines) == 0:
        logger.warning(f"No perpendicular lines found. raster_idx: {raster_idx}. raster_name: {raster_name}")
        # continue


    # ----- PLOTTING -----
    try:
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(21, 8))
        
        rgb_data = rasterio.open(tile_fp).read()[:3,:,:]
        rgb_data = rgb_data[::-1, :, :]
        rgb_data = (rgb_data-np.min(rgb_data))/(np.max(rgb_data)-np.min(rgb_data))

        # Plot: Raster + line + points + perpendiculars
        overlayed_pred = copy.deepcopy(rgb_data)
        overlayed_pred[2,:,:] = np.where(orig_mask, orig_mask, overlayed_pred[2,:,:])
        img1 = show(overlayed_pred, ax=ax1, title="Predicted water mask", extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax1.set_xlim(raster_bounds.left, raster_bounds.right)
        ax1.set_ylim(raster_bounds.bottom, raster_bounds.top)
        ax1.set_aspect(raster_aspect)
        ax1.axis("off")

        overlayed_segmentation = copy.deepcopy(rgb_data)
        overlayed_segmentation[2,:,:] = np.where(orig_mask, orig_mask, overlayed_segmentation[2,:,:])
        img1 = show(overlayed_segmentation, ax=ax2, title="Predicted water mask", extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax2.set_xlim(raster_bounds.left, raster_bounds.right)
        ax2.set_ylim(raster_bounds.bottom, raster_bounds.top)
        ax2.set_aspect(raster_aspect)
        nodes_gdf_raster.plot(ax=ax2, color='yellow', label='Node')
        line_gdf_raster.plot(ax=ax2, color='yellow', label='River reach')
        lines_gdf.plot(ax=ax2, color='yellow', linestyle='--', label='Pixel Counting Perpendicular Paths')
        # Annotate water pixel counts next to each point
        for idx, row in nodes_gdf.iterrows():
            ax2.annotate(str(row['width_m'])+"m", (row.geometry.x + 2, row.geometry.y + 2), color='yellow', size=25)
        # ax2.legend()
        ax2.axis("off")


        overlayed_segmentation2 = copy.deepcopy(rgb_data)
        overlayed_segmentation2[2,:,:] = np.where(raster_data, raster_data, overlayed_segmentation2[2,:,:])
        # Plot: Raster + line + points + perpendiculars
        img1 = show(overlayed_segmentation2, ax=ax3, title="Cleaned water mask with Reach, Nodes, and Perpendiculars", extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax3.set_xlim(raster_bounds.left, raster_bounds.right)
        ax3.set_ylim(raster_bounds.bottom, raster_bounds.top)
        ax3.set_aspect(raster_aspect)
        nodes_gdf_raster.plot(ax=ax3, color='yellow', label='Node')
        line_gdf_raster.plot(ax=ax3, color='yellow', label='River reach')
        lines_gdf.plot(ax=ax3, color='yellow', linestyle='--', label='Pixel Counting Perpendicular Paths')
        # Annotate water pixel counts next to each point
        for idx, row in nodes_gdf.iterrows():
            ax3.annotate(str(row['width_m'])+"m", (row.geometry.x + 2, row.geometry.y + 2), color='yellow', size=25)
        # ax3.legend()
        ax3.axis("off")

        plt.tight_layout()
        if args.is_gt==0:
            out_path_fig = os.path.join(actual_out_dir, f"{reach_id}_{raster_idx:03d}_{raster_name.replace('.tif','')}_node_widths.png")
        else:
            out_path_fig = os.path.join(actual_out_dir, f"{reach_id}_{raster_idx:03d}_{raster_name.replace('.tif','')}_node_widths_gt.png")
        logger.debug(f"Saving visualization to: {out_path_fig}")
        plt.savefig(out_path_fig, bbox_inches="tight", dpi=600)
        plt.close()
    except Exception as e:
        logger.error(f"{e}. raster_idx: {raster_idx}. raster_name: {raster_name}")


