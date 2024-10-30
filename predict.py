import os
import time
import numpy as np
import torch
import rasterio
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.HybridSN import HybridSN
from models.CNN_3D1D import CNN_3D_Classifer_1D
from utils.dataloader import *
from utils.utils import *
from utils.process import *

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "predict"
model_path = r"logs/2024-09-24-16-23-02-3D_1D_CNN-train-5/model_e004_loss1.6738.pt" # model path
# data settings
input_path = r"/home/ljs/HybridSN-pytorch/data/ZY1E_AHSI_E112.16_N22.37_20211114_011392_L1A0000363499-process.tif" # tif data
output_name = r"output5.tif" # tif data
# model settings
model_type = "3D_1D_CNN" 
patch_size = 5
num_classes = 8

if mode != "predict":
    raise ValueError("mode error!")

# training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

# read input data
img, im_geotrans, im_proj, cols, rows = read_tif(input_path)
print(f"input data shape: {img.shape}")
bands = img.shape[2]
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "HybridSN":
    time_folder = os.path.join("logs" ,time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + mode + "-" + str(patch_size))
    output_path = os.path.join(time_folder, output_name)
    os.makedirs(time_folder)
elif model_type == "3D_1D_CNN":
    time_folder = os.path.join("logs" ,time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + mode + "-" + str(patch_size))
    output_path = os.path.join(time_folder, output_name)
    os.makedirs(time_folder)
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# model initialized
#-------------------------------------------------------------------------------
if model_type == "HybridSN":
    model = HybridSN(
        in_channels = bands, 
        patch_size = patch_size, 
        num_classes = num_classes,
    )
elif model_type == "3D_1D_CNN":
    model = CNN_3D_Classifer_1D(
            input_channels = bands,
            num_classes = num_classes,
            patch_size = patch_size,
            dilation =  1
    )
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# preict
#-------------------------------------------------------------------------------
# padding image
padded_img = padding(img, patch_size)

# transform
transform = Compose([
    # Normalize(mean=0, std=1),
    # RandomHorizontalFlip(p=0.5),
])

# load weights
model = torch.load(model_path).to(device)
print(f"Load weights: {model_path}")
model.eval()

# initialized result
predictions = np.zeros((rows, cols))

# step of sliding
total_steps = rows * cols

# predict
with torch.no_grad():
    for x, y, patch in tqdm(sliding_window(padded_img, patch_size), total=total_steps, desc="Predicting"):
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        patch_tensor = transform(patch_tensor)
        output = model(patch_tensor)
        output = torch.argmax(output, dim=1)
        output = output.squeeze().cpu().numpy()
        predictions[x, y] = output

# store result
write_tif(output_path, np.expand_dims(predictions.astype('uint8'), axis=0), im_geotrans, im_proj)
#-------------------------------------------------------------------------------
