import torch
from torch.utils import data
from sklearn.model_selection import train_test_split

import cv2
import glob
import os



#READ DATA PATHS
def read_data_paths(main_path, n_data):
    """Create a list that store input and mask images [input_1, mask_1] ... [input_n, mask_n]. \n
        Args:
            - main_path (str): main path where data is stored.
            - n_data (int): number of data. \n
                0: all data
        """
    flairs_path = os.path.join(main_path, "data", "*", "data", "*_flair", "*.png")
    segs_path = os.path.join(main_path, "data", "*", "data", "*_seg", "*.png")

    data_list = []
    for flair_path, seg_path in zip(glob.glob(flairs_path), glob.glob(segs_path)):
        data_list.append([flair_path, seg_path])
        if len(data_list) == n_data:
            break
    return data_list



#TRAIN VAL TEST SPLIT
def train_val_test_split(dataset, test_val_split=0.15):
    """Split dataset into train, validation and test data. \n
        Args:
            - dataset (DataLoader): dataset to be splitted.
            - test_val_split (float): splitting ratio between 0 and 1.
                Default: 0.15
            """
    if test_val_split <= 0 or test_val_split >= 1:
        raise ValueError("Given test_val_split value ({}) causes ambiguity. It must be between 0 and 1.".format(test_val_split))

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_val_split)
    train_idx, test_idx = train_test_split(train_idx, test_size=test_val_split)
    datasets = {}
    
    datasets['train'] = data.Subset(dataset, train_idx)
    datasets['val'] = data.Subset(dataset, val_idx)
    datasets["test"] = data.Subset(dataset, test_idx)
    
    return datasets



#BRATS DATASET
class BratsDataset(data.Dataset):
    def __init__(self, main_path, n_data, transforms=None):
        """Creates brats dataset for segmentation tasks. \n
            Args:
                - main_path (str): path to the datas.
                - n_data (int): number of data.\n
                    0: all data
                - transforms (Transforms): transforms to be applied to the input and mask images.
                    Default: None
        """
        self.data_list = read_data_paths(main_path, n_data)
        self.transforms = transforms


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        img = cv2.imread(self.data_list[index][0], 0)
        mask = cv2.imread(self.data_list[index][1], 0)
        mask[mask == 4] = 3

        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).int().type("torch.LongTensor")

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask



#GENERAL DATASET
class GeneralDataSet(data.Dataset):
    def __init__(self, data_list, img_ch=1, mask_ch=1, transforms=None):
        """Creates dataset for segmentation tasks. \n
            Args:
                - data_list (list): path to the datas. data_list format should be [[img1, mask1], [img2, mask2], .... [imgn, maskn]]
                - img_ch (int): channels of input image     1: gray scale   3: rgb
                    Default: 1
                - mask_ch (int): channels of mask image     1: gray scale   3: rgb
                    Default: 1
                - transforms (Transforms): transforms to be applied to the input and mask images.
                    Default: None
        """
        self.data_list = data_list
        self.transforms = transforms
        self.img_ch = img_ch
        self.mask_ch = mask_ch
        

    def __len__(self):
        return len(self.data_list)

    
    def forward(self, index):
        if(self.img_ch == 1):
            img = cv2.imread(self.data_list[index][0], cv2.IMREAD_GRAYSCALE)
            img = torch.from_numpy(img).float().unsqueeze(0) / img.max()

        elif(self.img_ch == 3):
            img = cv2.imread(self.data_list[index][0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / img.max()
            img = torch.permute(img, (2, 0, 1))

        if(self.mask_ch == 1):
            mask = cv2.imread(self.data_list[index][1], cv2.IMREAD_GRAYSCALE)
            mask = torch.from_numpy(mask).int().unsqueeze(0)

        elif(self.mask_ch == 3):
            mask = cv2.imread(self.data_list[index][1])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = torch.from_numpy(mask).int()
            mask = torch.permute(mask, (2, 0, 1))

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)
        
        return img, mask
