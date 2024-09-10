# Dataset
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import os

def extract_center_patch(data, patch_size):
    """
    从每个样本中提取中心区域的补丁（patches）。
    
    Args:
        data (numpy.ndarray): 原始数据数组，形状为 [num, p, p, bands]。
        patch_size (int): 补丁的尺寸。
    
    Returns:
        numpy.ndarray: 提取后的数据数组，形状为 [num, patch_size, patch_size, bands]。
    """
    num, p, _, bands = data.shape
    # 计算中心区域的起始位置
    start = (p - patch_size) // 2
    end = start + patch_size
    
    # 提取中心区域
    center_patches = data[:, start:end, start:end, :]
    
    return center_patches

def load_and_split_data_by_file(data_file, label_file, train_ratio, val_ratio, patch_size=None):
    """
    读取 .mat 文件中的数据和标签，并根据比例切分为训练集、验证集和测试集，同时返回波段数量和标签类别数。
    
    Args:
        data_file (str): .mat 数据文件路径，文件中的数据应存储在 'data' 键中。
        label_file (str): .mat 标签文件路径，文件中的标签应存储在 'label' 键中。
        train_ratio (float): 训练集所占的比例，取值范围为 0 到 1。
        val_ratio (float): 验证集所占的比例，取值范围为 0 到 1。
        patch_size (int, optional): 补丁的尺寸。如果为 None，则不进行裁剪。默认是 None。
    
    Returns:
        tuple: 包含以下内容的元组
            - X_train (numpy.ndarray): 训练集数据，形状为 [num_train, patch_size, patch_size, bands]。
            - X_val (numpy.ndarray): 验证集数据，形状为 [num_val, patch_size, patch_size, bands]。
            - X_test (numpy.ndarray): 测试集数据，形状为 [num_test, patch_size, patch_size, bands]。
            - y_train (numpy.ndarray): 训练集标签，形状为 [num_train, 1]。
            - y_val (numpy.ndarray): 验证集标签，形状为 [num_val, 1]。
            - y_test (numpy.ndarray): 测试集标签，形状为 [num_test, 1]。
            - bands (int): 数据的波段数量（bands）。
            - num_classes (int): 标签的类别数量。
    """
    
    # 读取数据和标签
    data = sio.loadmat(data_file)['data']  # 假设数据存储在 'data' 键中
    labels = sio.loadmat(label_file)['label'].squeeze()  # 假设标签存储在 'label' 键中
    
    # 获取波段数量
    _, p, p, bands = data.shape

    if patch_size is not None:
        if p < patch_size:
            patch_size = None
    
    # 获取标签的类别数量
    num_classes = len(np.unique(labels))

    # 根据 patch_size 进行数据裁剪
    if patch_size is not None:
        data = extract_center_patch(data, patch_size)
    
    # 首先分割出训练集和剩余集
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=1-train_ratio, stratify=labels, random_state=42
    )
    
    # 然后从剩余集中分割出验证集和测试集
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_size, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, bands, num_classes

def load_and_split_data_by_folder(data_folder, label_folder, train_ratio, val_ratio, patch_size=None):
    """
    读取文件夹中的多个 .mat 文件的数据和标签，并根据比例切分为训练集、验证集和测试集，同时返回波段数量和标签类别数。
    
    Args:
        data_folder (str): 存储 .mat 数据文件的文件夹路径。
        label_folder (str): 存储 .mat 标签文件的文件夹路径。
        train_ratio (float): 训练集所占的比例，取值范围为 0 到 1。
        val_ratio (float): 验证集所占的比例，取值范围为 0 到 1。
        patch_size (int, optional): 补丁的尺寸。如果为 None，则不进行裁剪。默认是 None。
    
    Returns:
        tuple: 包含以下内容的元组
            - X_train (numpy.ndarray): 训练集数据，形状为 [num_train, patch_size, patch_size, bands]。
            - X_val (numpy.ndarray): 验证集数据，形状为 [num_val, patch_size, patch_size, bands]。
            - X_test (numpy.ndarray): 测试集数据，形状为 [num_test, patch_size, patch_size, bands]。
            - y_train (numpy.ndarray): 训练集标签，形状为 [num_train, 1]。
            - y_val (numpy.ndarray): 验证集标签，形状为 [num_val, 1]。
            - y_test (numpy.ndarray): 测试集标签，形状为 [num_test, 1]。
            - bands (int): 数据的波段数量（bands）。
            - num_classes (int): 标签的类别数量。
    """
    
    all_data = []
    all_labels = []

    # 遍历数据文件夹中的所有 .mat 文件
    data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.mat')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mat')])

    if len(data_files) != len(label_files):
        raise ValueError("数据文件和标签文件的数量不一致！")

    # 逐一读取每个数据文件和标签文件
    for data_file, label_file in zip(data_files, label_files):
        data_path = os.path.join(data_folder, data_file)
        label_path = os.path.join(label_folder, label_file)

        # 读取数据和标签
        data = sio.loadmat(data_path)['data']  # 假设数据存储在 'data' 键中
        labels = sio.loadmat(label_path)['label'].squeeze()  # 假设标签存储在 'label' 键中

        # 拼接所有数据
        all_data.append(data)
        all_labels.append(labels)

    # 将所有数据和标签拼接成一个大的数组
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 获取数据的形状信息
    m, p, p, bands = all_data.shape

    if patch_size is not None:
        if p < patch_size:
            patch_size = None

    # 获取标签的类别数量
    num_classes = len(np.unique(all_labels))

    # 根据 patch_size 进行数据裁剪
    if patch_size is not None:
        all_data = extract_center_patch(all_data, patch_size)

    # 首先分割出训练集和剩余集
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_data, all_labels, test_size=1 - train_ratio, stratify=all_labels, random_state=42
    )

    # 然后从剩余集中分割出验证集和测试集
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, bands, num_classes

class HSI_dataset(Dataset):
    def __init__(self, data_array, label_array, transform=None):
        """
        Args:
            data_array (numpy.ndarray): 形状为 [num, p, p, bands] 的数据数组。
            label_array (numpy.ndarray): 形状为 [num, 1] 的标签数组。
            transform (callable, optional): 应用于数据的变换。
        """
        self.data = data_array
        self.labels = label_array.squeeze() 
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]  # 形状为 [p, p, bands]
        data = torch.tensor(data).permute(2, 0, 1).float().unsqueeze(0)  # 转换为 [1, bands, p, p]
        label = torch.tensor(self.labels[index])  # 转换为 [1]

        if self.transform:
            data = self.transform(data)

        return data, label