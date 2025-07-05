import os, time
import numpy as np
from PIL import Image
from vis.WholeSlideImage import WholeSlideImage
from vis.wsi_utils import StitchCoords, StitchCoords_online


def visualize_prediction(bag_labels_patch, bag_predictions_patch, bag_coords, bag_wsi_names, wsi_path="./", vis_sample_interval=4):
    wsi_file_list = [file_name for file_name in os.listdir(wsi_path) if "xml" not in file_name.lower()]
    xml_file_list = [file_name for file_name in os.listdir(wsi_path) if "xml" in file_name.lower()]
    
    img_heatmaps = []
    label_heatmaps = []
    pred_heatmaps = []
    concat_heatmaps = []
    wsi_names = []
    
    sample_iterval = vis_sample_interval
    total_time=0
    for k, (label_patch, prediction_patch, coords, wsi_name) in enumerate(zip(bag_labels_patch, bag_predictions_patch, bag_coords, bag_wsi_names)):
        if k % sample_iterval != 0:
            continue 
        
        wsi_name = wsi_name[0]
        idx_list = [idx for idx in range(len(wsi_file_list)) if wsi_name in wsi_file_list[idx]]
        wsi_file_name = wsi_file_list[idx_list[0]]

        full_path = os.path.join(wsi_path, wsi_file_name)
        WSI_object = WholeSlideImage(full_path)
        
        label_patch = label_patch.squeeze(0).squeeze(-1)
        # prediction_patch = prediction_patch.numpy()
        img_heatmap, label_heatmap, pred_heatmap, concat_heatmap, elapsed_time = stitching(coords, WSI_object, label_patch, prediction_patch)
    
        img_heatmaps.append(img_heatmap)
        label_heatmaps.append(label_heatmap)
        pred_heatmaps.append(pred_heatmap)
        concat_heatmaps.append(concat_heatmap)
        wsi_names.append(wsi_name)
        total_time+=elapsed_time
    
    print(f"time lapse for visualizing; {total_time}")
    
    # label_heatmaps[0].save("test_label.png")
    # pred_heatmaps[0].save("test_pred.png")
    # img_heatmaps[0].save("test_img.png")
    
    return img_heatmaps, label_heatmaps, pred_heatmaps, concat_heatmaps, wsi_names

def stitching(coords, wsi_object, label_patch, prediction_patch, downscale = 128):
	start = time.time()
	img_heatmap, label_heatmap, pred_heatmap, concat_heatmap = StitchCoords_online(coords, wsi_object, label_patch, prediction_patch,\
                                                downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return img_heatmap, label_heatmap, pred_heatmap, concat_heatmap, total_time