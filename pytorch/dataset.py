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
    seg_paths = os.path.join(main_path, "*", "*_seg", "*.png")

    data_list = []
    for flair_path, seg_path in zip(glob.glob(flair_paths), glob.glob(seg_paths)):
        data_list.append([flair_path, seg_path])

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
        mask = cv2.imread(self.data_list[index][1], 0)
        mask[mask == 4] = 3

        flair = torch.from_numpy(flair).unsqueeze(0).float()
        mask = torch.from_numpy(mask).int().type("torch.LongTensor")

        if self.transforms is not None:
            flair = self.transforms(flair)
            mask = self.transforms(mask)

        return flair, mask