import torch
from torchvision import transforms
from torch.utils import data
from sklearn.model_selection import train_test_split

import cv2
import glob
import os



#READ DATA PATHS
def read_data_paths(main_path):
    """Create a list that store input and mask images [input_1, mask_1] ... [input_n, mask_n]. \n
        Args:
            - main_path (str): main path where data is stored.
            - n_data (int): number of data. \n
                0: all data
        """
    flair_paths = os.path.join(main_path, "*", "*_flair", "*.png")
    t1_paths = os.path.join(main_path, "*", "*_t1", "*.png")
    t1ce_paths = os.path.join(main_path, "*", "*t1ce", "*.png")
    t2_paths = os.path.join(main_path, "*", "*_t2", "*.png")
    seg_paths = os.path.join(main_path, "*", "*_seg", "*.png")

    data_list = []
    for flair_path, t1_path, t1ce_path, t2_path, seg_path in zip(sorted(glob.glob(flair_paths)), sorted(glob.glob(t1_paths)), sorted(glob.glob(t1ce_paths)), sorted(glob.glob(t2_paths)), sorted(glob.glob(seg_paths))):
        data_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path])

    return data_list

    

#TRAIN VAL TEST SPLIT
def train_val_test_split(dataset, test_val_split=0.15, shuffle=True):
    """Split dataset into train, validation and test data. \n
        Args:
            - dataset (DataLoader): dataset to be splitted.
            - test_val_split (float): splitting ratio between 0 and 1.
                Default: 0.15
    """
    if test_val_split <= 0 or test_val_split >= 1:
        raise ValueError("Given test_val_split value ({}) causes ambiguity. It must be between 0 and 1.".format(test_val_split))

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_val_split, shuffle=shuffle)
    train_idx, test_idx = train_test_split(train_idx, test_size=test_val_split, shuffle=shuffle)
    datasets = {}
    
    datasets['train'] = data.Subset(dataset, train_idx)
    datasets['val'] = data.Subset(dataset, val_idx)
    datasets["test"] = data.Subset(dataset, test_idx)
    
    return datasets



#BRATS DATASET
class BratsDataset(data.Dataset):
    def __init__(self, main_path, transforms=transforms.CenterCrop(160)):
        """Creates brats dataset for segmentation tasks. \n
            Args:
                - main_path (str): path to the datas.
                - transforms (Transforms): transforms to be applied to the input and mask images.
                    Default: None
        """
        self.data_list = read_data_paths(main_path)
        self.transforms = transforms


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        flair = cv2.imread(self.data_list[index][0], 0)
        t1 = cv2.imread(self.data_list[index][1], 0)
        t1ce = cv2.imread(self.data_list[index][2], 0)
        t2 = cv2.imread(self.data_list[index][3], 0)
        mask = cv2.imread(self.data_list[index][4], 0)
        mask[mask == 4] = 3

        flair = torch.from_numpy(flair).unsqueeze(0).float()
        t1 = torch.from_numpy(t1).unsqueeze(0).float()
        t1ce = torch.from_numpy(t1ce).unsqueeze(0).float()
        t2 = torch.from_numpy(t2).unsqueeze(0).float()

        img = torch.cat([flair, t1, t1ce, t2], dim=0)
        mask = torch.from_numpy(mask).int().type("torch.LongTensor")

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask