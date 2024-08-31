import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.HybridSN import HybridSN
from utils.dataloader import *
from utils.utils import *

def mirror_padding(image, patch_size):
    pad_size = patch_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    return padded_image

def sliding_window(image, patch_size):
    half_patch = patch_size // 2
    for y in range(half_patch, image.shape[0] - half_patch):
        for x in range(half_patch, image.shape[1] - half_patch):
            yield x, y, image[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "predict"
model_path = r"logs\2024-08-31-03-11-16-HybridSN-train\model.pt" # model path
# data settings
input_path = r"data\ZY1E_AHSI_E113.18_N22.37_20230131_917748_L1A0000563893_Process.tif" # tif data
output_name = r"output.tif" # tif data
# model settings
model_type = "HybridSN" 
patch_size = 33
num_classes = 2

if mode != "predict":
    raise ValueError("mode error!")

# training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read input data
img, im_geotrans, im_proj, cols, rows = read_tif(input_path)
print(f"input data shape: {img.shape}")
bands = img.shape[2]
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "HybridSN":
    time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + mode
    output_path = os.path.join(time_folder, output_name)
    os.makedirs(time_folder)
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# time setting
#-------------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
#-------------------------------------------------------------------------------

# model initialized
#-------------------------------------------------------------------------------
if model_type == "HybridSN":
    model = HybridSN(
        in_channels = bands, 
        patch_size = patch_size, 
        num_classes = num_classes,
    )
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# preict
#-------------------------------------------------------------------------------
model = model.to(device)

# padding image
padded_img = mirror_padding(img, patch_size)

# load weights
model = torch.load(model_path)
print(f"Load weights: {model_path}")
model.eval()

# initialized result
predictions = np.zeros((rows, cols))

# step of sliding
total_steps = (rows - patch_size + 1) * (cols - patch_size + 1)

# predict
with torch.no_grad():
    for x, y, patch in tqdm(sliding_window(padded_img, patch_size), total=total_steps, desc="Predicting"):
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        output = model(patch_tensor)
        output = torch.argmax(output, dim=1)
        output = output.squeeze().cpu().numpy()
        predictions[x, y] = output

# store result
write_tif(output_path, np.expand_dims(predictions, axis=0), im_geotrans, im_proj, gdal.GDT_Byte)
#-------------------------------------------------------------------------------
