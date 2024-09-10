import os
import rasterio
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import shape, Point
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import random

# ------------------------------ 重采样函数 ------------------------------
def resample_tif(input_path, output_path, scale_factor=30):
    print(f"正在重采样文件: {input_path}")
    with rasterio.open(input_path) as dataset:
        transform, width, height = calculate_default_transform(
            dataset.crs, dataset.crs, dataset.width, dataset.height, 
            *dataset.bounds, dst_width=dataset.width // scale_factor, dst_height=dataset.height // scale_factor
        )
        
        kwargs = dataset.meta.copy()
        kwargs.update({
            'crs': dataset.crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(dataset, 1),
                destination=rasterio.band(dst, 1),
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=transform,
                dst_crs=dataset.crs,
                resampling=Resampling.nearest
            )
    print(f"重采样完成，输出文件: {output_path}")

# ------------------------------ 将TIF文件转换为点矢量文件 ------------------------------
def tif_to_point_vector(tif_path, vector_path):
    print(f"正在将TIF文件转换为点矢量文件: {tif_path}")
    with rasterio.open(tif_path) as src:
        image = src.read(1)  # 读取第一波段数据
        transform = src.transform
        
        results = []
        for (i, j), value in np.ndenumerate(image):
            if value != 0:
                # 获取像元中心点坐标
                x, y = transform * (j + 0.5, i + 0.5)
                point = Point(x, y)
                results.append({'geometry': point, 'label': value})
        
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(results, crs=src.crs, geometry='geometry')
    gdf.to_file(vector_path, driver="ESRI Shapefile")
    print(f"转换完成，输出文件: {vector_path}")

# ------------------------------ 合并矢量文件 ------------------------------
def merge_and_clip_vectors(vector_folder, output_vector, clip_shapefile):
    print(f"正在合并矢量文件到: {output_vector}")

    # 读取裁剪范围的 Shapefile
    clip_gdf = gpd.read_file(clip_shapefile)
    print(f"裁剪范围 CRS: {clip_gdf.crs}")

    # 初始化存放矢量数据的列表
    gdf_list = []
    for file in os.listdir(vector_folder):
        if file.endswith(".shp"):
            gdf = gpd.read_file(os.path.join(vector_folder, file))
            print(f"读取文件: {file}")

            # 转换为 WGS84 坐标系 (EPSG:4326)
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            gdf_list.append(gdf)
    
    # 合并所有矢量数据
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs='EPSG:4326')
    print("矢量数据合并完成")

    # 确保裁剪范围和合并后的数据使用相同的坐标系
    if merged_gdf.crs != clip_gdf.crs:
        print(f"合并数据 CRS: {merged_gdf.crs} 与裁剪数据 CRS 不匹配，转换裁剪数据 CRS")
        clip_gdf = clip_gdf.to_crs(merged_gdf.crs)
        print(f"裁剪数据 CRS 转换为: {merged_gdf.crs}")

    # 对合并后的数据进行裁剪
    clipped_gdf = gpd.clip(merged_gdf, clip_gdf)
    print("裁剪操作完成")

    # 将裁剪后的数据保存到输出文件
    clipped_gdf.to_file(output_vector, driver="ESRI Shapefile")
    print(f"裁剪后的矢量文件输出完成，输出文件: {output_vector}")

# ------------------------------ 列出目录下所有文件 ------------------------------
def list_files_in_directory(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# ------------------------------ 主程序 ------------------------------
def main(input_folder, output_folder, final_vector, clip_vector):
    print("开始处理...")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resampled_folder = os.path.join(output_folder, "resampled_tifs")
    vector_folder = os.path.join(output_folder, "vectors")

    if not os.path.exists(resampled_folder):
        os.makedirs(resampled_folder)
    if not os.path.exists(vector_folder):
        os.makedirs(vector_folder)

    # 重采样每个tif文件
    for file in os.listdir(input_folder):
        if file.lower().endswith(".tif"):  # 忽略扩展名大小写
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(resampled_folder, file)
            resample_tif(input_file, output_file)

    # 将重采样后的tif转换为矢量文件
    for file in os.listdir(resampled_folder):
        if file.lower().endswith(".tif"):  # 忽略扩展名大小写
            input_file = os.path.join(resampled_folder, file)
            output_file = os.path.join(vector_folder, file.replace(".tif", ".shp").replace(".TIF", ".shp"))
            tif_to_point_vector(input_file, output_file)

    # 合并所有矢量文件
    merge_and_clip_vectors(vector_folder, final_vector, clip_vector)
    print("处理完成！")

# ------------------------------ 随机筛选点数据 ------------------------------
def random_sample_shapefile(input_shapefile, output_shapefile, n):
    """
    从输入的 shapefile 文件中随机筛选出 n 个点，并将结果保存到输出的 shapefile 文件中。
    
    :param input_shapefile: 输入的 shapefile 文件路径
    :param output_shapefile: 输出的 shapefile 文件路径
    :param n: 要筛选的点的数量
    """
    # 读取输入的 shapefile 文件
    gdf = gpd.read_file(input_shapefile)
    
    # 确保 n 不超过点的总数
    if n > len(gdf):
        raise ValueError("指定的点的数量 n 超过了 shapefile 中的点的总数")
    
    # 随机抽取 n 个点
    sampled_gdf = gdf.sample(n=n, random_state=42)
    
    # 将结果保存为新的 shapefile 文件
    sampled_gdf.to_file(output_shapefile)
    print(f"从{input_shapefile}随机筛选{str(n)}个点写出至{output_shapefile}")


# ------------------------------ 点数据处理与生成patches ------------------------------
# def process_raster_and_points(points_shapefile, raster_directory, patch_output_path, label_output_path, P=33, valid_ratio=0.5):
#     # 读取点shapefile
#     points_gdf = gpd.read_file(points_shapefile)
#     print("Points CRS:", points_gdf.crs)

#     # 多个栅格文件路径列表
#     raster_files = list_files_in_directory(raster_directory)

#     # 存放所有patch的列表
#     all_patches = []
#     # 存放所有标签的列表
#     all_labels = []

#     # 遍历每个栅格文件并处理
#     for raster_index, raster_file in enumerate(raster_files):
#         print(f"Processing raster file {raster_index + 1}/{len(raster_files)}: {raster_file}")
        
#         with rasterio.open(raster_file) as raster:
#             transform = raster.transform
#             raster_crs = raster.crs
#             raster_height, raster_width = raster.height, raster.width
#             band_count = raster.count  # 获取波段数
            
#             # 打印栅格的 CRS
#             print("Raster CRS:", raster_crs)

#             # 将点数据的 CRS 转换为当前栅格的 CRS
#             if points_gdf.crs != raster_crs:
#                 transformed_points_gdf = points_gdf.to_crs(raster_crs)
#                 print("Points CRS transformed to:", raster_crs)
#             else:
#                 transformed_points_gdf = points_gdf

#             # 计算点的行列号
#             transformed_points_gdf['row'], transformed_points_gdf['col'] = ~transform * (transformed_points_gdf.geometry.x, transformed_points_gdf.geometry.y)
#             transformed_points_gdf['row'] = transformed_points_gdf['row'].astype(int)
#             transformed_points_gdf['col'] = transformed_points_gdf['col'].astype(int)

#             # 创建一个掩模来标记有效数据区域
#             valid_mask = np.ones((raster_height, raster_width), dtype=bool)
#             for band in range(band_count):
#                 band_data = raster.read(band + 1)
#                 valid_mask &= (band_data != 0)  # 假设0是黑色/无效值

#             print("Valid mask created. Checking mask values...")
#             print("Mask area with valid data:", np.sum(valid_mask))

#             for index, row in transformed_points_gdf.iterrows():
#                 point_row = row['row']
#                 point_col = row['col']
                
#                 # 计算patch的范围
#                 min_row = point_row - P // 2
#                 max_row = point_row + P // 2 + 1
#                 min_col = point_col - P // 2
#                 max_col = point_col + P // 2 + 1
                
#                 # 检查patch是否完全在栅格范围内
#                 if (0 <= min_row < raster_height and
#                     0 <= max_row <= raster_height and
#                     0 <= min_col < raster_width and
#                     0 <= max_col <= raster_width):
                    
#                     # 提取patch
#                     patch = np.zeros((P, P, band_count), dtype=np.float32)
#                     patch_valid = np.zeros((P, P), dtype=bool)
                    
#                     for band in range(band_count):
#                         band_data = raster.read(band + 1)
#                         patch[:, :, band] = band_data[min_row:max_row, min_col:max_col]
#                         patch_valid |= (band_data[min_row:max_row, min_col:max_col] != 0)  # 标记有效数据区域
                    
#                     # 计算有效数据比例
#                     valid_data_ratio = np.sum(patch_valid) / (P * P)
                    
#                     if valid_data_ratio > valid_ratio:  # 假设至少50%的有效数据
#                         # 添加到patch列表
#                         all_patches.append(patch)
                        
#                         # 提取标签
#                         label = row['label']
#                         all_labels.append(label)
                        
#                         print(f"Extracted patch and label for point index {index}")

#     # 保存patches为MAT文件
#     if all_patches:
#         all_patches_array = np.array(all_patches)  # 转换为numpy数组
#         print(f"Final patch array shape: {all_patches_array.shape}")  # 打印形状
#         savemat(patch_output_path, {'data': all_patches_array})
#         print(f"All patches have been saved to {patch_output_path}")
        
#         # 将所有标签保存为一个.mat文件
#         all_labels_array = np.array(all_labels).reshape(-1, 1)  # 转换为numpy数组并重塑为 (patch数量, 1)
#         print(f"Final labels array shape: {all_labels_array.shape}")  # 打印形状
#         savemat(label_output_path, {'label': all_labels_array})
#         print(f"All labels have been saved to {label_output_path}")
#     else:
#         print("No valid patches were extracted.")

# ------------------------------ 点数据处理与生成patches ------------------------------
def process_raster_and_points(points_shapefile, raster_directory, patch_output_path, label_output_path, P=33, valid_ratio=0.5, n=1000):
    # 读取点shapefile
    points_gdf = gpd.read_file(points_shapefile)
    print("Points CRS:", points_gdf.crs)

    # 多个栅格文件路径列表
    raster_files = list_files_in_directory(raster_directory)

    # 存放所有patch的列表
    all_patches = []
    # 存放所有标签的列表
    all_labels = []
    
    # 用于文件命名
    file_counter = 1

    # 遍历每个栅格文件并处理
    for raster_index, raster_file in enumerate(raster_files):
        print(f"Processing raster file {raster_index + 1}/{len(raster_files)}: {raster_file}")
        
        with rasterio.open(raster_file) as raster:
            transform = raster.transform
            raster_crs = raster.crs
            raster_height, raster_width = raster.height, raster.width
            band_count = raster.count  # 获取波段数
            
            # 打印栅格的 CRS
            print("Raster CRS:", raster_crs)

            # 将点数据的 CRS 转换为当前栅格的 CRS
            if points_gdf.crs != raster_crs:
                transformed_points_gdf = points_gdf.to_crs(raster_crs)
                print("Points CRS transformed to:", raster_crs)
            else:
                transformed_points_gdf = points_gdf

            # 计算点的行列号
            transformed_points_gdf['row'], transformed_points_gdf['col'] = ~transform * (transformed_points_gdf.geometry.x, transformed_points_gdf.geometry.y)
            transformed_points_gdf['row'] = transformed_points_gdf['row'].astype(int)
            transformed_points_gdf['col'] = transformed_points_gdf['col'].astype(int)

            # 创建一个掩模来标记有效数据区域
            valid_mask = np.ones((raster_height, raster_width), dtype=bool)
            for band in range(band_count):
                band_data = raster.read(band + 1)
                valid_mask &= (band_data != 0)  # 假设0是黑色/无效值

            print("Valid mask created. Checking mask values...")
            print("Mask area with valid data:", np.sum(valid_mask))

            for index, row in transformed_points_gdf.iterrows():
                point_row = row['row']
                point_col = row['col']
                
                # 计算patch的范围
                min_row = point_row - P // 2
                max_row = point_row + P // 2 + 1
                min_col = point_col - P // 2
                max_col = point_col + P // 2 + 1
                
                # 检查patch是否完全在栅格范围内
                if (0 <= min_row < raster_height and
                    0 <= max_row <= raster_height and
                    0 <= min_col < raster_width and
                    0 <= max_col <= raster_width):
                    
                    # 提取patch
                    patch = np.zeros((P, P, band_count), dtype=np.float32)
                    patch_valid = np.zeros((P, P), dtype=bool)
                    
                    for band in range(band_count):
                        band_data = raster.read(band + 1)
                        patch[:, :, band] = band_data[min_row:max_row, min_col:max_col]
                        patch_valid |= (band_data[min_row:max_row, min_col:max_col] != 0)  # 标记有效数据区域
                    
                    # 计算有效数据比例
                    valid_data_ratio = np.sum(patch_valid) / (P * P)
                    
                    if valid_data_ratio > valid_ratio:  # 假设至少50%的有效数据
                        # 添加到patch列表
                        all_patches.append(patch)
                        
                        # 提取标签
                        label = row['label']
                        all_labels.append(label)
                        
                        print(f"Extracted patch and label for point index {index}")
                        
                        # 每当数量达到 n，保存当前数据到.mat文件并重置列表
                        if len(all_patches) >= n:
                            all_patches_array = np.array(all_patches)
                            all_labels_array = np.array(all_labels).reshape(-1, 1)
                            
                            patch_file = f"{os.path.splitext(patch_output_path)[0]}_{file_counter}.mat"
                            label_file = f"{os.path.splitext(label_output_path)[0]}_{file_counter}.mat"
                            
                            savemat(patch_file, {'data': all_patches_array})
                            savemat(label_file, {'label': all_labels_array})
                            
                            print(f"Saved {len(all_patches)} patches to {patch_file} and labels to {label_file}")
                            
                            # 重置列表和计数器
                            all_patches = []
                            all_labels = []
                            file_counter += 1

    # 保存剩余的patches和labels
    if all_patches:
        all_patches_array = np.array(all_patches)  # 转换为numpy数组
        print(f"Final patch array shape: {all_patches_array.shape}")  # 打印形状
        final_patch_file = f"{os.path.splitext(patch_output_path)[0]}_{file_counter}.mat"
        savemat(final_patch_file, {'data': all_patches_array})
        print(f"Remaining patches have been saved to {final_patch_file}")
        
        # 将所有标签保存为一个.mat文件
        all_labels_array = np.array(all_labels).reshape(-1, 1)  # 转换为numpy数组并重塑为 (patch数量, 1)
        print(f"Final labels array shape: {all_labels_array.shape}")  # 打印形状
        final_label_file = f"{os.path.splitext(label_output_path)[0]}_{file_counter}.mat"
        savemat(final_label_file, {'label': all_labels_array})
        print(f"Remaining labels have been saved to {final_label_file}")
    else:
         print("No valid patches were extracted.")
              

def sample_patches_by_class(data_path, label_path, patches_output_path, labels_output_path, n_per_class):
    # 读取MAT文件
    data = loadmat(data_path)
    label = loadmat(label_path)
    all_patches = data['data']
    all_labels = data['label']
    
    # 将标签转换为列表
    all_labels = all_labels.flatten()
    
    # 获取所有标签的唯一值
    unique_labels = np.unique(all_labels)
    
    # 存储每个类的patches和labels
    class_patches = {}
    class_labels = {}
    
    for label in unique_labels:
        class_patches[label] = []
        class_labels[label] = []

    # 按标签将patches和labels分类
    for patch, label in zip(all_patches, all_labels):
        class_patches[label].append(patch)
        class_labels[label].append(label)
    
    final_patches = []
    final_labels = []
    
    # 随机选择每个类的n个样本
    for label, patches in class_patches.items():
        if len(patches) > n_per_class:
            selected_indices = np.random.choice(len(patches), n_per_class, replace=False)
            selected_patches = [patches[i] for i in selected_indices]
            selected_labels = [class_labels[label][i] for i in selected_indices]
        else:
            selected_patches = patches
            selected_labels = class_labels[label]
        
        final_patches.extend(selected_patches)
        final_labels.extend(selected_labels)
    
    # 保存选定的patches和labels为MAT文件
    savemat(patches_output_path, {'data': np.array(final_patches)})
    savemat(labels_output_path, {'label': np.array(final_labels)})
    print(f"Sampled patches saved to {patches_output_path}")
    print(f"Sampled labels saved to {labels_output_path}")

# ------------------------------ 程序入口 ------------------------------
if __name__ == "__main__":
    input_folder = r"E:\202405\20240509hyper\samples\GDGQ_img+samples\2019label"
    output_folder = r"H:\project\202406\20240610hyper\data\label_shp"
    clip_vector = r"H:\project\202406\20240610hyper\data\range\cities.shp"
    final_vector = os.path.join(output_folder, "label.shp")
    
    # 调用主程序
    main(input_folder, output_folder, final_vector, clip_vector)
    
    # 调用点数据处理函数
    points_shapefile = final_vector
    random_shapefile = os.path.join(output_folder, "random_label.shp")
    raster_directory = r"H:\project\202406\20240610hyper\data\2021"
    patch_output_path = r"H:\project\202406\20240610hyper\data\samples\data_patches.mat"
    label_output_path = r"H:\project\202406\20240610hyper\data\samples\label_patches.mat"


    random_sample_shapefile(points_shapefile, random_shapefile, 10000)

    process_raster_and_points(random_shapefile, raster_directory, patch_output_path, label_output_path)

    # 调用采样函数
    # selected_patch_output_path = r"H:\project\202406\20240610hyper\data\samples\selected_data_patches.mat"
    # selected_label_output_path = r"H:\project\202406\20240610hyper\data\samples\selected_label_patches.mat"
     
    # sample_patches_by_class(patch_output_path, label_output_path, selected_patch_output_path, selected_label_output_path, n_per_class=100)
