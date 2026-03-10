from PIL import Image
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import warnings
import torch
import glob
import random
# 设置 numpy 打印选项（全局生效）
np.set_printoptions(
    threshold=np.inf,  # 禁用截断，显示所有元素
    linewidth=np.inf   # 不限制每行字符数（根据终端宽度自动换行）
)

# 固定随机种子确保可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
#############################根据图像名称获得各部分信息
def parse_filename(filename):
    # 去除文件扩展名并按'_'分割
    parts = filename.split('.png')[0].split('_')
    # print(parts)
    
    # 提取各部分信息
    dataset_id = parts[0]
    temperature = int(parts[1].replace('k', ''))
    composition_str = parts[2]
    ph_time = int(parts[3])
    crop_id = parts[-1]
    
    # 处理成分信息
    x, y = map(float, composition_str.split('-'))
    composition = [round(1 - x - y, 3), x, y]  # 保留两位小数
    
    return {
        'dataset_id': dataset_id,
        'temperature': temperature,
        'composition': composition,
        'ph_time': ph_time,
        'crop_id': crop_id,
        'description':filename
    }

# # 测试示例
# filename = "01_1473k_0.165-0.055_60_crop_01.png"
# result = parse_filename(filename)
# print(result['composition'])

#######################################

def crop_and_save_image_grid(image_path, output_dir, block_size=256, stride=128):
    """
    从图像左上角开始按固定步长裁剪图像块，并保存到指定目录
    :param image_path: 输入图像路径
    :param output_dir: 输出目录路径
    :param block_size: 裁剪块大小（默认128）
    :param stride: 滑动步长（默认128，重叠）
    :return: 保存的文件路径列表
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始图像基本名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 打开原始图像
    img = Image.open(image_path)
    width, height = img.size
    saved_paths = []
    count = 1  # 计数器从1开始

    # 遍历行列进行裁剪
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # 计算裁剪区域
            box = (x, y, x + block_size, y + block_size)
            
            # 跳过超出边界的块
            if box[2] > width or box[3] > height:
                continue
                
            # 执行裁剪
            patch = img.crop(box)
            
            # 生成保存路径
            save_name = f"{base_name}_crop_{count:02d}.png"
            save_path = os.path.join(output_dir, save_name)
            
            # 保存图像
            patch.save(save_path)
            saved_paths.append(save_path)
            count += 1
# crop_and_save_image_grid("/home/yons/code/PhaseSimuDiffusionModel/data_of_512/01_1473k_0.160-0.050_60.png",'/home/yons/code/PhaseSimuDiffusionModel/data_crop_256_time_60', block_size=256, stride=64)
# png_files = glob.glob(os.path.join("/home/yons/code/PhaseSimuDiffusionModel/sim60_data_of_512", "*_60.png"))
# # 328
# print("要裁剪的图片数量",len(png_files))
# for path in png_files:
#     crop_and_save_image_grid(path,'/home/yons/code/PhaseSimuDiffusionModel/data_crop_256_time_60', block_size=256, stride=64)

#########################NOTE 最终采用考虑平衡各个条件的样本数量的随机裁剪方式
from collections import defaultdict
import re

def dynamic_crop(source_dir, output_dir, target_total=50, block_size=256, random_seed=42):
    """
    动态裁剪处理以_60.png结尾的图像
    
    参数:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        target_total: 每组总裁剪数(默认50)
        block_size: 裁剪尺寸(默认256)
        random_seed: 随机种子(默认42)
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 获取所有以_60.png结尾的图像文件
    img_files = sorted(glob.glob(os.path.join(source_dir, "*_60.png")))
    if not img_files:
        print(f"警告: 在 {source_dir} 中未找到符合*_60.png格式的图像!")
        return
    
    # 2. 按条件前缀分组
    condition_groups = defaultdict(list)
    for file_path in img_files:
        # 提取文件名并分组
        filename = os.path.basename(file_path)
        condition_key = re.sub(r"^\d{2}_", "", filename)  # 移除数字前缀
        condition_groups[condition_key].append(file_path)
    # 对分组键和组内文件排序
    sorted_conditions = sorted(condition_groups.keys())
    for key in sorted_conditions:
        condition_groups[key] = sorted(condition_groups[key])
    global_crop_count=0
    # 3. 对每组进行处理
    for condition_key in sorted_conditions:
        file_list = condition_groups[condition_key]
        num_origin = len(file_list)
        # 计算每张原图应裁剪的数量
        per_image = target_total // num_origin
        extra_crops = target_total % num_origin
        # 对组内每张图像进行裁剪
        for idx, img_path in enumerate(sorted(file_list)):
            # 创建裁剪计数器
            crop_count = 1
            crops_needed = per_image + (1 if idx < extra_crops else 0)
            # 处理单张图像
            img = Image.open(img_path)
            w, h = img.size
            
            # 进行随机裁剪
            for crop_idx in range(crops_needed):
                # 生成随机位置
                x = random.randint(0, w - block_size)
                y = random.randint(0, h - block_size)
                
                # 执行裁剪
                box = (x, y, x + block_size, y + block_size)
                patch = img.crop(box)
                
                # 创建保存路径 (使用全局计数器)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{base_name}_crop_{crop_count:02d}.png"
                save_path = os.path.join(output_dir, save_name)
                
                # 保存图像
                patch.save(save_path)
                crop_count += 1
                global_crop_count+=1
    
    print(f"处理完成! 共生成 {global_crop_count} 张裁剪图像")

#################################最新的图片裁剪
# source_dir = "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/sim60_data_of_512"
# output_dir = "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60"
# #处理完成! 共生成 11150 张裁剪图像
# dynamic_crop(source_dir, output_dir)
#########################################验证裁剪后的图像形状

# image_path_1 = "/home/yons/code/PhaseSimuDiffusionModel/data_crop_256_time_60/02_1473k_0.160-0.050_60_crop_01.png"
# image_path_2="/home/yons/code/PhaseSimuDiffusionModel/data_of_512/02_1473k_0.160-0.060_60.png"
# img_1 = Image.open(image_path_1)
# arr_1 = np.array(img_1)
# img_2 = Image.open(image_path_2)
# arr_2 = np.array(img_2)
# print("#############验证裁剪后的形状######################")
# print(arr_1.shape)
# print(img_1.mode)  
# print(arr_2.shape)
# print(img_2.mode)  # 单通道输出应为 "L"

####################################划分验证集和测试集

def partition_by_compositions(full_img_dir, target_compositions, val_fraction=0.1):
    """
    按多个指定化学成分划分数据集
    
    参数：
        full_img_dir: 完整数据集路径
        target_compositions: 多个目标测试成分的列表 
            (如: [[0.7, 0.15, 0.15], [0.6, 0.2, 0.2]])
        val_fraction: 验证集比例 (非目标成分的)
    
    返回：
        train_set, val_set, test_set: 划分好的数据集子集
    """
    from torch.utils.data import Subset
    from collections import defaultdict
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # 1. 将每个目标成分转换为唯一键
    target_comp_keys = set()
    for comp in target_compositions:
        # 确保所有成分都是标准三元组
        assert len(comp) == 3, "每个目标成分必须是三元组"
        comp_key = tuple(np.round(comp, 3))
        target_comp_keys.add(comp_key)
    
    # 2. 收集所有样本信息
    composition_to_indices = defaultdict(list)
    # 遍历数据集收集成分信息
    # png_files = glob.glob(os.path.join(full_img_dir, "*.png"))
    # 只获取文件名列表
    png_files = [os.path.basename(file_path) 
                for file_path in glob.glob(os.path.join(full_img_dir, "*.png"))]
    for i,path in enumerate(png_files):
        result=parse_filename(path)
        composition=tuple(np.round(result['composition'], 3))
        composition_to_indices[composition].append(i)
    # 3. 标识所有目标成分样本
    test_indices = []
    found_comps = set()
    missing_comps = set()
    
    for comp_key in target_comp_keys:
        if comp_key in composition_to_indices:
            test_indices.extend(composition_to_indices[comp_key])
            found_comps.add(comp_key)
        else:
            missing_comps.add(comp_key)
    
    # 4. 报告缺失成分信息
    if missing_comps:
        print(f"警告：找不到以下成分的样本: {list(missing_comps)}")  
    
    # 5. 去重并创建测试集
    test_indices = list(set(test_indices))
    
    # 6. 创建非目标成分数据集
    all_indices = set(range(len(png_files)))
    non_target_indices = list(all_indices - set(test_indices))
    
    # 7. 划分训练和验证集
    train_indices, val_indices = train_test_split(
        non_target_indices,
        test_size=val_fraction,
        random_state=42
    )
    
    # 8. 创建子集
    train_set = Subset(png_files, train_indices)
    val_set = Subset(png_files, val_indices)
    test_set = Subset(png_files, test_indices)
    
    # 9. 生成详细报告
    # print("\n 数据集划分报告")
    # print(f"  目标成分数量: {len(target_compositions)}")
    # print(f"  匹配目标成分: {len(found_comps)}")
    # # 测试集样本: 1000 (9.0%)
    # print(f"  测试集样本: {len(test_set)} ({len(test_set)/len(png_files)*100:.1f}%)")
    # # 训练集样本: 9135 (81.9%)
    # print(f"  训练集样本: {len(train_set)} ({len(train_set)/len(png_files)*100:.1f}%)")
    # # 验证集样本: 1015 (9.1%)
    # print(f"  验证集样本: {len(val_set)} ({len(val_set)/len(png_files)*100:.1f}%)")
    
    return train_set, val_set, test_set
# #20个测试成分
# test_com=[[0.735, 0.16, 0.105],[0.74, 0.16, 0.1],[0.745,0.16,0.095],[0.75, 0.16,0.09],[0.755, 0.16, 0.085],[0.76, 0.16, 0.08],
#           [0.765, 0.16, 0.075],[0.77, 0.16, 0.07],[0.775, 0.16, 0.065],[0.78, 0.16, 0.06],[0.785, 0.16, 0.055],[0.79, 0.16, 0.05],
#           [0.75, 0.165, 0.085],[0.799, 0.166, 0.035],[0.72, 0.17, 0.11],[0.765, 0.175, 0.06],[0.754, 0.176, 0.07],[0.804, 0.176, 0.02],
#           [0.779,0.181,0.04],[0.784, 0.206, 0.01]]
# train_set_0,val_set_0,test_set_0=partition_by_compositions("/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", test_com)


##################将训练集、验证集和测试集从总数据集中分离出来
import shutil
from pathlib import Path
import concurrent.futures

def copy_val_images(val_set, source_dir, target_dir):
    """复制验证集图片到新目录
    
    Args:
        val_set: 数据集中对应的文件名称，可以是一个列表
        source_dir: 原数据集目录
        target_dir: 目标保存目录
    """
    # 创建目标目录（若不存在）
    target_path = Path(target_dir)
    # 如果目标目录已存在且非空，则清空
    if target_path.exists() and any(target_path.iterdir()):
        print(f"⚠️ 目标目录非空，正在清空: {target_dir}")
        # 删除目录下所有内容
        for item in target_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    # 确保目录存在
    target_path.mkdir(parents=True, exist_ok=True)
    # 更高效获取文件名的实现
    img_names=list(val_set) 
    
    # 准备复制任务列表
    copy_tasks = []
    for name in img_names:
        src_path = Path(source_dir) / name
        dst_path = Path(target_dir) / name
        
        # 检查源文件是否存在
        if not src_path.exists():
            print(f"⚠️ 文件不存在: {src_path}")
            continue
            
        copy_tasks.append((src_path, dst_path))
    
    # 使用多线程并行复制（加快大文件复制速度）
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for src, dst in copy_tasks:
            futures.append(executor.submit(shutil.copy2, src, dst))
        
        # 等待所有复制完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ 复制失败: {e}")

    print(f"✅ 成功复制 {len(copy_tasks)} 张图片到 {target_dir}")


###################执行分离数据集 
# # 1015张
# copy_val_images(val_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
#                 "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/val_data_crop_256_time_60")
# # 9135张
# copy_val_images(train_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
#                 "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/train_data_crop_256_time_60")
# # 1000张
# copy_val_images(test_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
#                 "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/test_data_crop_256_time_60")
##############数据增强和数据加载器定义
import torchvision.transforms as transforms

def augmentation_pipeline(train_data=True,resolution=(256,256)):
    if train_data:
        augmentation_pipeline = transforms.Compose([
            # 水平/垂直翻转二选一（互斥）
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=1.0),  # 强制水平翻转
                    transforms.RandomVerticalFlip(p=1.0)     # 强制垂直翻转
                ])
            ], p=0.5),  # 整体有50%概率执行翻转
            
            # 固定角度三选一旋转（90°, 180°, 270°）
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=(90, 90)),
                    transforms.RandomRotation(degrees=(180, 180)),
                    transforms.RandomRotation(degrees=(270, 270))
                ])
            ], p=0.5),
            # 已经是目标尺寸时，不会进行插值操作
            transforms.Resize(
                size=resolution,                # 目标尺寸（H, W）
                # 双线性插值
                interpolation=transforms.InterpolationMode.BILINEAR  # 插值方法
                # interpolation=transforms.InterpolationMode.NEAREST
            ),
            # 张量转换 + 数值归一化到[-1,1]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 映射公式：(x-0.5)/0.5
        ])
        
    else:
        augmentation_pipeline = transforms.Compose([
            # resize放在数据增强后
            transforms.Resize(
                size=resolution,                # 目标尺寸（H, W）
                # 双线性插值
                interpolation=transforms.InterpolationMode.BILINEAR  # 插值方法
                # interpolation=transforms.InterpolationMode.NEAREST
            ),
            # 张量转换 + 数值归一化到[-1,1]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 映射公式：(x-0.5)/0.5
        ])
    return augmentation_pipeline




img_files = [
            f for f in os.listdir('/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/sim60_data_of_512/')
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

composition_list=[]
#print(len(img_files))
for i,path in enumerate(img_files):
    result=parse_filename(path)
    #print(result['composition'])
    composition_list.append(result['composition'])
    

#print(len(composition_list))
composition_tensor=torch.tensor(composition_list)
#print(composition_tensor.shape)

#print(composition_tensor)
unique_compositions = torch.unique(
    composition_tensor,
    dim=0,                # 按行去重
    return_inverse=False,  # 返回原始数据到唯一数据的映射
    sorted=False          # 不排序以保持原始顺序
)
# torch.Size([234, 3])
# print(unique_compositions.shape)
#print(unique_compositions)



# 数据集类（加载预处理后的图片并动态增强）
class PhaseSimulDataset(Dataset):
    def __init__(self, img_dir, resolution=(256, 256),train_data=True):
        self.img_dir = img_dir
         # 1. 过滤图像文件
        self.img_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.img_files:
            warnings.warn(f"No valid images found in {img_dir}", UserWarning)
        self.transform = augmentation_pipeline(train_data=train_data,resolution=resolution)
        
        

    def __getitem__(self, idx):
        img_file_name=self.img_files[idx]
        img_description=parse_filename(img_file_name)
        img_path = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_path).convert('L')     # 灰度图加载
        if self.transform:
            img = self.transform(img)
        img_description['image']=img
        return img_description
    
    def __len__(self):
        return len(self.img_files)
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数，支持混合类型字段"""
        # 过滤无效样本（如加载失败返回None的情况）
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        # 自动识别字段类型并处理
        collated_batch = {}
        
        # 处理图像张量（假设已经过ToTensor）
        if 'image' in batch[0]:
            collated_batch['image'] = torch.stack(
                [item['image'] for item in batch], 
                dim=0
            )

        # 处理数值型字段
        numeric_fields = ['temperature', 'ph_time']
        for field in numeric_fields:
            if field in batch[0]:
                collated_batch[field] = torch.tensor(
                    [item[field] for item in batch],
                    dtype=torch.int32  # 温度和时间用整型
                )

        # 处理浮点数组字段
        if 'composition' in batch[0]:
            collated_batch['composition'] = torch.tensor(
                [item['composition'] for item in batch],
                dtype=torch.float32
            )

        # 处理字符串字段（保持列表格式）
        string_fields = ['dataset_id', 'crop_id', 'description']
        for field in string_fields:
            if field in batch[0]:
                collated_batch[field] = [item[field] for item in batch]

        return collated_batch


# NOTE 不同的训练阶段要改下面的参数设置正确的batch_size
input_image_shape=256
train_latent_ddim_or_valid_autoencoder=True
# 图像大小等于256时，批量只能设置为8，16不行
# VAE+unet潜在扩散模型训练时或VAE验证时批量都能设置为64，速度更快，但是VAE+DiT_S_1跑不动
# batch_size=64 if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else 8
# 4090的设置如下
# batch_size=32 if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else 8
# a800，显存80g，可以增大批量，但发现效果不好，减小批量以增加参数更新次数
# batch_size=128 if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else 32
# a800的设置，和4090的全局批量保持一致
batch_size=64 if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else 32
print(f"----------batch_size of training is {batch_size}")
# 创建数据加载器
train_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/train_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=True)
val_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/val_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=False)
test_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/test_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=False)

# 调整验证/测试集的批次大小（更大以加速评估）
# TODO DiT_S_1使用128的批量在生成验证集图像时会显存不足
# val_batch_size = min(128, len(val_set)) if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else min(32,len(val_set)) 
# DiT_S_1验证集的批量设置为128跑不动，设置为64，DiT_S_2设置为128跑到399 epochs时会内存不足，也需要设置为64
# val_batch_size = min(64, len(val_set)) if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else min(32,len(val_set))
# a800增大批量
val_batch_size = min(256, len(val_set)) if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else min(128,len(val_set))
test_batch_size = min(25, len(test_set))  # 测试集可用更小批次
#########TODO 先去除以debug: pin_memory=True,persistent_workers=True, prefetch_factor=None
train_dataloader=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None, collate_fn=PhaseSimulDataset.collate_fn,drop_last=True)  # 避免反复创建worker
# #######验证集可以都设置大一点
val_dataloader=DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None,collate_fn=PhaseSimulDataset.collate_fn,drop_last=False)  
test_dataloader=DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None,collate_fn=PhaseSimulDataset.collate_fn,drop_last=False)  

def get_img_shape():
    return (1, input_image_shape, input_image_shape)



######################获取训练集、验证集、测试集的成分
train_coms = [data["composition"] for data in train_set]
train_composition_tensor=torch.tensor(train_coms)

train_unique_compositions, counts = torch.unique(
    train_composition_tensor,
    dim=0,
    return_counts=True,
    return_inverse=False,  # 不返回映射
    sorted=False
)

# 直接构建字典
train_per_com_imgs_count_dict = {
    tuple([round(x.item(), 3) for x in comp]): cnt.item() 
    for comp, cnt in zip(train_unique_compositions, counts)
}
# # torch.Size([203, 3])
print("训练集中成分数量",train_unique_compositions.shape)
# print("训练集中成分图片数量计数",train_per_com_imgs_count_dict)
# 验证取出某个成分的数量
# cond=train_unique_compositions[0]
# cond_key = tuple(round(x.item(), 3) for x in cond)
# target_count = train_per_com_imgs_count_dict.get(cond_key)
# print(f'{cond_key} 的样本数量为 {target_count}')
##############验证集的
val_coms = [data["composition"] for data in val_set]
val_composition_tensor=torch.tensor(val_coms)

val_unique_compositions = torch.unique(
    val_composition_tensor,
    dim=0,                # 按行去重
    return_inverse=False,  # 返回原始数据到唯一数据的映射
    sorted=False          # 不排序以保持原始顺序
)
# 验证集中成分 torch.Size([202, 3])
print("验证集中成分",val_unique_compositions.shape)
#########测试集
test_coms = [data["composition"] for data in test_set]
test_composition_tensor=torch.tensor(test_coms)
test_unique_compositions = torch.unique(
    test_composition_tensor,
    dim=0,                # 按行去重
    return_inverse=False,  # 返回原始数据到唯一数据的映射
    sorted=False          # 不排序以保持原始顺序
)
# torch.Size([22, 3])
# print("测试集成分",test_unique_compositions)

if __name__=='__main__':
    #9135
    print(train_set.__len__())
    # print(len(train_set))
    print(train_set[0]['composition'])
    print(val_set[0]["description"])
    # 验证批次数据结构
    sample_batch = next(iter(val_dataloader))
    # torch.Size([32, 1, 128, 128])
    print(f"图像张量维度: {sample_batch['image'].shape}")        # [B,C,H,W]
    print(f"温度字段类型: {sample_batch['temperature'].dtype}")   # torch.int32
    # 成分字段示例: torch.Size([32, 3])
    print(f"成分字段形状: {sample_batch['composition'].shape}")      # 浮点张量
    print(f"成分字段示例: {sample_batch['composition'].dtype}")      # 浮点张量
    print(f"描述字段类型: {type(sample_batch['description'][0])}") # <class 'str'>
    # 图片名称
    print(f"描述字段示例: {sample_batch['description'][0]}")
    ##########################划分数据集
    # # copy_val_images(val_set, "PhaseSimuDiffusionModel/data_crop_256_time_60", "PhaseSimuDiffusionModel/val_data_crop_256_time_60")
    # copy_val_images(train_set, "PhaseSimuDiffusionModel/data_crop_256_time_60", "PhaseSimuDiffusionModel/train_data_crop_256_time_60")
    # copy_val_images(test_set, "PhaseSimuDiffusionModel/data_crop_256_time_60", "PhaseSimuDiffusionModel/test_data_crop_256_time_60")
    