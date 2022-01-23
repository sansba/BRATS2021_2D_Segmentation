import torch
from torch.utils import data
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np

import cv2

import tarfile
import patoolib as ptl

import glob
import os



#Extract Main File > OK
def extract_tar(from_path, to_path):
    #Create path if it does not exist
    if not os.path.isdir(to_path):
        os.makedirs(to_path, exist_ok=True)

    #TAR file extracting process
    print("Extracting Data...")
    tar = tarfile.open(from_path)
    tar.extractall(to_path)
    tar.close()



#Extract GZ Files > OK
def extract_gz(path):
    #Path finding process
    for path1 in glob.glob(path + "/*"):
        for path2_f, path2_s in zip(glob.glob(path1 + "/*_flair*"), glob.glob(path1 + "/*_seg*")):
            #Out direction of flair and seg
            out_dir_f = path2_f.split(".")[0]
            out_dir_s = path2_s.split(".")[0]

            #Create out directions
            os.mkdir(out_dir_f)
            os.mkdir(out_dir_s)

            #Extract corresponding data to output path
            ptl.extract_archive(path2_f, outdir=out_dir_f)
            ptl.extract_archive(path2_s, outdir=out_dir_s)




#Create Image > OK
def create_nii_slices(path, to):
    #Find nii files of flair and seg
    for path1 in glob.glob(path + "/*"):
        for path2_f, path2_s in zip(glob.glob(path1 + "/*_flair"), glob.glob(path1 + "/*_seg")):
            for path3_f, path3_s in zip(glob.glob(path2_f + "/*_flair*"), glob.glob(path2_s + "/*_seg*")):
                #Path for output
                patient_num = path1.split("/")[-1]
                flair_name = patient_num + "_flair"
                seg_name = patient_num + "_seg"

                #Create output paths if it does not exist
                if not os.path.exists(os.path.join(to, patient_num, flair_name)):
                    os.makedirs(os.path.join(to, patient_num, flair_name))
                    os.makedirs(os.path.join(to, patient_num, seg_name))
                
                #Read nii files
                flair = nib.load(path3_f).get_fdata()
                seg = nib.load(path3_s).get_fdata()

                #Create image files
                for i in range(flair.shape[2]):
                    if (seg[:, :, i] == 1).any() or (seg[:, :, i] == 2).any() or (seg[:, :, i] == 4).any():
                        
                        #Turn images into integer
                        flair_img = flair[:, :, i]
                        flair_img = np.uint8(flair_img / flair_img.max() * 255)
                        seg_img = seg[:, :, i]
                        seg_img = np.uint8(seg_img)

                        #Flair and seg paths
                        flair_path = os.path.join(to, patient_num, flair_name, str(i) + ".png")
                        seg_path = os.path.join(to, patient_num, seg_name, str(i) + ".png")

                        #Create images
                        cv2.imwrite(flair_path, flair_img)
                        cv2.imwrite(seg_path, seg_img)




#Read images path as a list > OK
def read_data_paths(main_path):
    flairs_path = os.path.join(main_path, "data", "*", "data", "*_flair", "*.png")
    segs_path = os.path.join(main_path, "data", "*", "data", "*_seg", "*.png")

    data_list = []
    for flair_path, seg_path in zip(glob.glob(flairs_path), glob.glob(segs_path)):
        data_list.append([flair_path, seg_path])
    return data_list



#Splitting data into three parts
def train_val_test_dataset(dataset, test_val_split=0.15):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_val_split)
    train_idx, test_idx = train_test_split(train_idx, test_size=test_val_split)
    datasets = {}
    

    datasets['train'] = data.Subset(dataset, train_idx)
    datasets['val'] = data.Subset(dataset, val_idx)
    datasets["test"] = data.Subset(dataset, test_idx)
    
    return datasets



#Create Dataset
class BratsDataset(data.Dataset):
    def __init__(self, main_path, transforms=None):
        self.data_list = read_data_paths(main_path)
        self.transforms = transforms


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        flair = cv2.imread(self.data_list[idx][0], 0)
        seg = cv2.imread(self.data_list[idx][1], 0)
        seg[seg == 4] = 3

        flair = torch.from_numpy(flair).unsqueeze(0).float()
        seg = torch.from_numpy(seg).int().type("torch.LongTensor")

        if self.transforms is not None:
            flair = self.transforms(flair)
            seg = self.transforms(seg)

        return flair, seg
