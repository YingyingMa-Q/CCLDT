from PIL import Image
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import warnings
import torch
import glob
import random
np.set_printoptions(
    threshold=np.inf,  
    linewidth=np.inf   
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
############################# Extract metadata/information from the image filename.
def parse_filename(filename):
    parts = filename.split('.png')[0].split('_')
    dataset_id = parts[0]
    temperature = int(parts[1].replace('k', ''))
    composition_str = parts[2]
    ph_time = int(parts[3])
    crop_id = parts[-1]
    
    x, y = map(float, composition_str.split('-'))
    composition = [round(1 - x - y, 3), x, y]  
    return {
        'dataset_id': dataset_id,
        'temperature': temperature,
        'composition': composition,
        'ph_time': ph_time,
        'crop_id': crop_id,
        'description':filename
    }



import re
def extract_gen_com_from_filename(filename):
    """
    Extract the first three float-point compositions from filenames formatted like 'comp_0.720_0.170_0.110_...'.
    Args:
        filename (str): The input filename string.
        
    Returns:
        list: A list containing three extracted floats; returns an empty list if no match is found.
    """
    pattern = r"comp_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)"
    match = re.search(pattern, filename)
    if match:
        from pathlib import Path
        filename = Path(filename)
        composition =list(map(float, match.groups()))
        return {
        'composition': composition,
        'description':filename.stem
    }
    else:
        return []

#######################################
from collections import defaultdict
import re

def dynamic_crop(source_dir, output_dir, target_total=50, block_size=256, random_seed=42):
    """
    Dynamic cropping for images ending with '_60.png'.
    Args:
        source_dir (str): Path to the source directory.
        output_dir (str): Path to the output directory.
        target_total (int): Total number of crops per image group (default: 50).
        block_size (int): Size of the crop window (default: 256).
        random_seed (int): Random seed for reproducibility (default: 42).
    """
    random.seed(random_seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(source_dir, "*_60.png")))
    if not img_files:
        print(f"警告: 在 {source_dir} 中未找到符合*_60.png格式的图像!")
        return
    
    condition_groups = defaultdict(list)
    for file_path in img_files:
        filename = os.path.basename(file_path)
        condition_key = re.sub(r"^\d{2}_", "", filename)  
        condition_groups[condition_key].append(file_path)
    sorted_conditions = sorted(condition_groups.keys())
    for key in sorted_conditions:
        condition_groups[key] = sorted(condition_groups[key])
    global_crop_count=0
    for condition_key in sorted_conditions:
        file_list = condition_groups[condition_key]
        num_origin = len(file_list)
        per_image = target_total // num_origin
        extra_crops = target_total % num_origin
        for idx, img_path in enumerate(sorted(file_list)):
            crop_count = 1
            crops_needed = per_image + (1 if idx < extra_crops else 0)
            img = Image.open(img_path)
            w, h = img.size
            for crop_idx in range(crops_needed):
                x = random.randint(0, w - block_size)
                y = random.randint(0, h - block_size)
                
                # crop
                box = (x, y, x + block_size, y + block_size)
                patch = img.crop(box)
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{base_name}_crop_{crop_count:02d}.png"
                save_path = os.path.join(output_dir, save_name)
                
                # saving images
                patch.save(save_path)
                crop_count += 1
                global_crop_count+=1
    
    print(f"处理完成! 共生成 {global_crop_count} 张裁剪图像")

#################################cropping images
# Check if the dataset has been cropped (e.g., verify if the output directory exists).
data_crop_dir = "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60"
dataset_exists = os.path.exists(data_crop_dir)
if not dataset_exists:
    source_dir = "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/sim60_data_of_512"
    dynamic_crop(source_dir, data_crop_dir)

####################################

def partition_by_compositions(full_img_dir, target_compositions, val_fraction=0.1):
    """
    Split the dataset based on multiple specified chemical compositions.

    Args:
        full_img_dir (str): Path to the full dataset directory.
        target_compositions (list[list[float]]): A list of target compositions for the test set. 
            (e.g., [[0.7, 0.15, 0.15], [0.6, 0.2, 0.2]])
        val_fraction (float): Fraction of the remaining data (non-target compositions) 
            to be used for the validation set.

    Returns:
        tuple: (train_set, val_set, test_set) containing the partitioned dataset subsets.
    """
    from torch.utils.data import Subset
    from collections import defaultdict
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    
    target_comp_keys = set()
    for comp in target_compositions:
        assert len(comp) == 3, "每个目标成分必须是三元组"
        comp_key = tuple(np.round(comp, 3))
        target_comp_keys.add(comp_key)
    
    composition_to_indices = defaultdict(list)
    png_files = [os.path.basename(file_path) 
                for file_path in glob.glob(os.path.join(full_img_dir, "*.png"))]
    for i,path in enumerate(png_files):
        result=parse_filename(path)
        composition=tuple(np.round(result['composition'], 3))
        composition_to_indices[composition].append(i)
    test_indices = []
    found_comps = set()
    missing_comps = set()
    
    for comp_key in target_comp_keys:
        if comp_key in composition_to_indices:
            test_indices.extend(composition_to_indices[comp_key])
            found_comps.add(comp_key)
        else:
            missing_comps.add(comp_key)
    
    if missing_comps:
        print(f"警告：找不到以下成分的样本: {list(missing_comps)}")  
    
    # construct test set
    test_indices = list(set(test_indices))
    
    all_indices = set(range(len(png_files)))
    non_target_indices = list(all_indices - set(test_indices))
    
    train_indices, val_indices = train_test_split(
        non_target_indices,
        test_size=val_fraction,
        random_state=42
    )
    
    train_set = Subset(png_files, train_indices)
    val_set = Subset(png_files, val_indices)
    test_set = Subset(png_files, test_indices)
    
    return train_set, val_set, test_set


##################Split the full dataset into train, validation, and test subsets.
import shutil
from pathlib import Path
import concurrent.futures

def copy_val_images(val_set, source_dir, target_dir):
    """
    Split dataset images into specified directories.

    Args:
        val_set (list): A list of filenames corresponding to the validation set.
        source_dir (str): Path to the source dataset directory.
        target_dir (str): Path to the target saving directory.
    """
    # Create target directory (if it doesn't exist).
    target_path = Path(target_dir)
    # If the target directory exists and is not empty, clear its contents.
    if target_path.exists() and any(target_path.iterdir()):
        print(f"⚠️ 目标目录非空，正在清空: {target_dir}")
        for item in target_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    target_path.mkdir(parents=True, exist_ok=True)
    img_names=list(val_set) 
    
    copy_tasks = []
    for name in img_names:
        src_path = Path(source_dir) / name
        dst_path = Path(target_dir) / name
        
        if not src_path.exists():
            print(f"⚠️ 文件不存在: {src_path}")
            continue
            
        copy_tasks.append((src_path, dst_path))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for src, dst in copy_tasks:
            futures.append(executor.submit(shutil.copy2, src, dst))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ 复制失败: {e}")

    print(f"✅ 成功复制 {len(copy_tasks)} 张图片到 {target_dir}")


###################partitioning datasets
train_dir = "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/train_data_crop_256_time_60"
dataset_exists = os.path.exists(train_dir)
if not dataset_exists:
    #20 unseen compositions
    test_com=[[0.735, 0.16, 0.105],[0.74, 0.16, 0.1],[0.745,0.16,0.095],[0.75, 0.16,0.09],[0.755, 0.16, 0.085],[0.76, 0.16, 0.08],
            [0.765, 0.16, 0.075],[0.77, 0.16, 0.07],[0.775, 0.16, 0.065],[0.78, 0.16, 0.06],[0.785, 0.16, 0.055],[0.79, 0.16, 0.05],
            [0.75, 0.165, 0.085],[0.799, 0.166, 0.035],[0.72, 0.17, 0.11],[0.765, 0.175, 0.06],[0.754, 0.176, 0.07],[0.804, 0.176, 0.02],
            [0.779,0.181,0.04],[0.784, 0.206, 0.01]]
    train_set_0,val_set_0,test_set_0=partition_by_compositions("/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", test_com)
    # 1015张
    copy_val_images(val_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
                    "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/val_data_crop_256_time_60")
    # 9135张
    copy_val_images(train_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
                    "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/train_data_crop_256_time_60")
    # 1000张
    copy_val_images(test_set_0, "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/data_crop_256_time_60", 
                    "/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/test_data_crop_256_time_60")
##############Data augmentation and DataLoader definitions.
import torchvision.transforms as transforms

def augmentation_pipeline(train_data=True,resolution=(256,256)):
    if train_data:
        augmentation_pipeline = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=1.0),  
                    transforms.RandomVerticalFlip(p=1.0)     
                ])
            ], p=0.5),  # Apply random flip with a 50% probability.
            
            # Random rotation from fixed angles (90°, 180°, 270°).
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=(90, 90)),
                    transforms.RandomRotation(degrees=(180, 180)),
                    transforms.RandomRotation(degrees=(270, 270))
                ])
            ], p=0.5),
            transforms.Resize(
                size=resolution,                # target res（H, W）
                interpolation=transforms.InterpolationMode.BILINEAR  
            ),
            # Tensor conversion and normalization to [-1, 1].
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # formula：(x-0.5)/0.5
        ])
        
    else:
        augmentation_pipeline = transforms.Compose([
            # resize after augmentation
            transforms.Resize(
                size=resolution,                # target res（H, W）
                interpolation=transforms.InterpolationMode.BILINEAR  
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # formula：(x-0.5)/0.5
        ])
    return augmentation_pipeline




img_files = [
            f for f in os.listdir('/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/sim60_data_of_512/')
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

composition_list=[]
for i,path in enumerate(img_files):
    result=parse_filename(path)
    composition_list.append(result['composition'])
    
composition_tensor=torch.tensor(composition_list)

unique_compositions = torch.unique(
    composition_tensor,
    dim=0,                
    return_inverse=False,  
    sorted=False          
)

# Dataset class (loads pre-processed images and applies on-the-fly augmentation).
class PhaseSimulDataset(Dataset):
    def __init__(self, img_dir, resolution=(256, 256),train_data=True,gen_data=False):
        self.img_dir = img_dir
        self.img_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.img_files:
            warnings.warn(f"No valid images found in {img_dir}", UserWarning)
        self.transform = augmentation_pipeline(train_data=train_data,resolution=resolution)
        self.gen_data=gen_data
        

    def __getitem__(self, idx):
        img_file_name=self.img_files[idx]
        if not self.gen_data:
            img_description=parse_filename(img_file_name)
        else:
            img_description=extract_gen_com_from_filename(img_file_name)
        img_path = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_path).convert('L')     
        if self.transform:
            img = self.transform(img)
        img_description['image']=img
        return img_description
    
    def __len__(self):
        return len(self.img_files)
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function supporting mixed-type fields."""
        # Filter out invalid samples (e.g., cases where loading fails and returns None).
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        collated_batch = {}
        
        # Handle image tensors.
        if 'image' in batch[0]:
            collated_batch['image'] = torch.stack(
                [item['image'] for item in batch], 
                dim=0
            )

        # Handle numerical fields.
        numeric_fields = ['temperature', 'ph_time']
        for field in numeric_fields:
            if field in batch[0]:
                collated_batch[field] = torch.tensor(
                    [item[field] for item in batch],
                    dtype=torch.int32  
                )

        # Handle float array fields.
        if 'composition' in batch[0]:
            collated_batch['composition'] = torch.tensor(
                [item['composition'] for item in batch],
                dtype=torch.float32
            )

        # Handle string fields (preserve as a list).
        string_fields = ['dataset_id', 'crop_id', 'description']
        for field in string_fields:
            if field in batch[0]:
                collated_batch[field] = [item[field] for item in batch]

        return collated_batch


input_image_shape=256
# Set to True for DiT training or ACC-AE validation; set to False for ACC-AE training. This ensures the appropriate batch_size is applied for each stage.
train_latent_ddim_or_valid_autoencoder=True
batch_size=64 if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else 32
print(f"----------batch_size of training is {batch_size}")
# Create data loaders.
train_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/train_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=True)
val_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/val_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=False)
test_set=PhaseSimulDataset(img_dir='/work/home/acldoa64po/code/PhaseSimuDiffusionModel/time60_cond_ddpm/data/test_data_crop_256_time_60/',resolution=(input_image_shape,input_image_shape),train_data=False)

val_batch_size = min(256, len(val_set)) if input_image_shape==128 or train_latent_ddim_or_valid_autoencoder else min(128,len(val_set))
test_batch_size = min(25, len(test_set))  
train_dataloader=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None, collate_fn=PhaseSimulDataset.collate_fn,drop_last=True)  
val_dataloader=DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None,collate_fn=PhaseSimulDataset.collate_fn,drop_last=False)  
test_dataloader=DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True, prefetch_factor=None,collate_fn=PhaseSimulDataset.collate_fn,drop_last=False)  

def get_img_shape():
    return (1, input_image_shape, input_image_shape)



######################Get compositions for training, validation, and test sets.
train_coms = [data["composition"] for data in train_set]
train_composition_tensor=torch.tensor(train_coms)

train_unique_compositions, counts = torch.unique(
    train_composition_tensor,
    dim=0,
    return_counts=True,
    return_inverse=False,  
    sorted=False
)

# construct counts dict for each training composition
train_per_com_imgs_count_dict = {
    tuple([round(x.item(), 3) for x in comp]): cnt.item() 
    for comp, cnt in zip(train_unique_compositions, counts)
}
print("训练集中成分数量",train_unique_compositions.shape)

##############validation set
val_coms = [data["composition"] for data in val_set]
val_composition_tensor=torch.tensor(val_coms)

val_unique_compositions = torch.unique(
    val_composition_tensor,
    dim=0,                
    return_inverse=False,  
    sorted=False          
)

#########test set
test_coms = [data["composition"] for data in test_set]
test_composition_tensor=torch.tensor(test_coms)
test_unique_compositions = torch.unique(
    test_composition_tensor,
    dim=0,                
    return_inverse=False, 
    sorted=False          
)


