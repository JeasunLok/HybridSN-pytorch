import torch
import torch.nn as nn
import torch.nn.functional as F
# from osgeo import gdal
import rasterio
import numpy as np

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class FocalLoss(nn.Module):  
    def __init__(self, alpha=0.5, gamma=2, reduction='mean', ignore_index=0):  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.ignore_index = ignore_index  
        self.reduction = reduction  
  
    def forward(self, inputs, targets):  
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)  
        pt = torch.exp(-ce_loss)  
        focal_term = (1 - pt) ** self.gamma  
        focal_loss = self.alpha * focal_term * ce_loss  
  
        if self.reduction == 'mean':  
            loss = focal_loss.mean()  
        elif self.reduction == 'sum':  
            loss = focal_loss.sum()  
        else:   
            loss = focal_loss  
  
        return loss  
#-------------------------------------------------------------------------------
    
# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

# transform
#-------------------------------------------------------------------------------
# 自定义归一化 transform
class Normalize:
    def __init__(self, mean=0.0, std=1.0):
        """
        初始化归一化类。

        参数:
        target_mean (float): 目标均值。
        target_std (float): 目标标准差。
        """
        self.target_mean = mean
        self.target_std = std

    def __call__(self, sample):
        """
        对输入样本进行归一化到指定的均值和标准差。

        参数:
        sample (ndarray): 输入的样本。

        返回:
        归一化后的样本。
        """
        current_mean = torch.mean(sample)
        current_std = torch.std(sample)
        
        if current_std == 0:
            # 避免除以零的情况
            return torch.full_like(sample, self.target_mean)

        normalized_sample = (sample - current_mean) / current_std
        return normalized_sample * self.target_std + self.target_mean

# 自定义水平翻转 transform
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() < self.p:
            sample = torch.flip(sample, dims=[2])
        return sample

# 组合多个变换的 Compose 类
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
#-------------------------------------------------------------------------------
    

#-------------------------------------------------------------------------------
def log_training_results(file_path, mode, epoch=None, train_loss=None, train_acc=None,
                         OA_val=None, AA_val=None, Kappa_val=None, CA_val=None, CM_val=None,
                         OA_test=None, AA_test=None, Kappa_test=None, CA_test=None, CM_test=None,
                         epoch_num=None):
    with open(file_path, 'a') as log_file:
        if mode == "train":
            if epoch_num is not None:
                log_file.write(f"Epoch: {epoch_num:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}%\n")
            
            if OA_val is not None and AA_val is not None and Kappa_val is not None and CA_val is not None and CM_val is not None:
                log_file.write("===============================================================================\n")
                log_file.write(f"Epoch: {epoch_num:03d} => Validation Results:\n")
                log_file.write(f"OA: {OA_val*100:.4f}% | AA: {AA_val*100:.4f}% | Kappa: {Kappa_val:.4f}\n")
                log_file.write("CA: ")
                log_file.write(f"{CA_val}\n")
                log_file.write("Validation Confusion Matrix:\n")
                log_file.write(f"{CM_val}\n")
                log_file.write("===============================================================================\n")
        
        elif mode == "test":
            if OA_test is not None and AA_test is not None and Kappa_test is not None and CA_test is not None and CM_test is not None:
                log_file.write("===============================================================================\n")
                log_file.write("Test Results:\n")
                log_file.write(f"OA: {OA_test*100:.4f}% | AA: {AA_test*100:.4f}% | Kappa: {Kappa_test:.4f}\n")
                log_file.write("CA: ")
                log_file.write(f"{CA_test}\n")
                log_file.write("Test Confusion Matrix:\n")
                log_file.write(f"{CM_test}\n")
                log_file.write("===============================================================================\n")
        
        if epoch_num is None:
            log_file.write("End of process\n")
            log_file.write("===============================================================================\n")
#-------------------------------------------------------------------------------
               
# 读取tif
def read_tif(path):
    with rasterio.open(path) as dataset:
        im_data = dataset.read()  # 读取数据
        if len(im_data) == 2:  # 处理单波段情况
            im_data = im_data[np.newaxis, :, :]  # 添加波段维度
        im_data = np.transpose(im_data, [1, 2, 0])  # 转置为 (height, width, channels)
        im_proj = dataset.crs  # 读取投影
        im_geotrans = dataset.transform  # 读取仿射变换
        cols, rows = dataset.width, dataset.height
    return im_data, im_geotrans, im_proj, cols, rows


# 写出tif
def write_tif(newpath, im_data, im_geotrans, im_proj):
    if len(im_data) == 2:  # 处理二维数据的情况
        im_data = im_data[np.newaxis, :, :]  # 添加一个波段维度
    bands = im_data.shape[0]
    height = im_data.shape[1]
    width = im_data.shape[2]
    datatype = im_data.dtype  # 获取数据类型

    with rasterio.open(newpath, 'w', driver='GTiff', height=height, 
                       width=width, count=bands, 
                       dtype=datatype, crs=im_proj, transform=im_geotrans) as new_dataset:
        for i in range(bands):
            new_dataset.write(im_data[i, :, :], i + 1)
