import glob
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class SeisDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, image_size=256):
        """
        Initialize the custom dataset.
        Args:
            file_paths (list): List of data-file paths.
            transform (callable, optional): Data-preprocessing function.
        """
        self.exclude_keywords = list(range(950,1000))

        if split == 'train':
            self.file_paths = [
                os.path.join(root, file) for file in os.listdir(root)
                if os.path.isfile(os.path.join(root, file))  # Include files only, excluding subdirectories.
                and int(file.split('_')[0]) not in self.exclude_keywords # Filter files by the specified identifiers.
            ]
        else:
            self.file_paths = [
                os.path.join(root, file) for file in os.listdir(root)
                if os.path.isfile(os.path.join(root, file))  # Include files only, excluding subdirectories.
                and int(file.split('_')[0]) in self.exclude_keywords  # Filter files by the specified identifiers.
            ]
        print(split,' dataset:', len(self.file_paths))
        self.transforms = transform
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load the sample at the requested index on demand.
        """
        file_path = self.file_paths[idx]
        # Load data from the file; adapt the loading logic if needed.
        data = torch.load(file_path)
        base_name = os.path.basename(file_path)
        
        imp_min = eval(base_name.split('_')[1])
        imp_max = eval(base_name.split('_')[2])
        seis_min = eval(base_name.split('_')[3])
        seis_max = eval(base_name.split('_')[4])
        dt = eval(base_name.split('_')[5]) / 1000
        f0 = eval(base_name.split('_')[6])
        pos = base_name.split('_')[7].split('.')[0]
        
        image = data['imp'].unsqueeze(0).to(torch.float32)
        cond = data['seis'].unsqueeze(0).to(torch.float32)
        
        # interval = 1
        # max_num = 3
        # width = 2 * interval + 1
        # rdm = torch.randint(0, image.shape[1], (1 + max_num,))
        # # rdm2 = torch.randint(0, (image.shape[1] // width) - 1, (max_num,))
        # rdm2 = torch.randperm((image.shape[1] // width) - 1)[:max_num]
        # if int(base_name.split('_')[0]) in self.exclude_keywords:
        #     N = (rdm[0] / (image.shape[1] / (max_num))).int() + 1
        # else:
        #     N = (rdm[0] / (image.shape[1] / (max_num + 1))).int()
        # F = (rdm[1:4] / (image.shape[1] / width)).int()
        # pos = (rdm2 * width + F).clamp(0, image.shape[1]).int()
        
        if self.transforms:
            image, cond = self.transforms(image, cond)
            
        # well = torch.zeros_like(image) 
        # pos_f = (pos[:N].unsqueeze(1) + torch.arange(width)).flatten()
        # # well[:,pos_f] = image[:,pos[:N]+interval].unsqueeze(2).repeat(1,1,width,1).reshape(1,-1,image.shape[-1])
        # well[:,pos_f] = image[:,pos_f].view(image.shape[0], N, width, image.shape[-1]).mean(2,keepdim=True).repeat(1,1,width,1).view(image.shape[0],-1,image.shape[-1])
        
        # print(file_path, f0)
        return {'image':image, 'mask':cond , 'dt':dt, 'f0':f0, 'imp_min':imp_min, 'imp_max':imp_max, 'seis_min':seis_min, 'seis_max':seis_max}  
 


if __name__ == '__main__':
    root= '../ImageNet2012/'
    cond_info_path = os.path.join(root, 'cond_info.json')
    if os.path.exists(cond_info_path):
        print('load ImageNetC from json')
        with open(cond_info_path, 'r') as file:
            cond_info = json.load(file)
            mask_paths = cond_info['mask']
            canny_paths = cond_info['canny']
            depth_paths = cond_info['depth']
            normal_paths = cond_info['normal']

    from tqdm import tqdm
    with tqdm(total=len(normal_paths)) as pbar:
        for path in normal_paths:
            try:
                img = Image.open(path)#.convert('RGB')
            except:
                print(path)
            # img_path = path.replace('normal', 'train').replace('.json', '.JPEG').replace('.jpeg', '.JPEG')
            #     with open(mask_path, 'r') as f:
            #         mask_info = json.load(f)
            #     mask = process_anns(mask_info, 512, colormap).astype(np.uint8)  # 512 is fixed during the labelling
            #     mask = Image.fromarray(mask)
            #     mask.save('mask.png')
            #     print(mask.path)
            # except:
            #     fail_list.append(mask_path)
            #     print(mask_path)

            pbar.update(1)
    # for path in fail_list:
    #     os.system(f'rm {path}')
