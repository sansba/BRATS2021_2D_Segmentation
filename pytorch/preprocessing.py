import os
import glob

import tarfile
import patoolib as ptl

import numpy as np
import cv2
import nibabel as nib



#EXTRACT TAR FILE
def extract_tar_file(from_path: str, to_path: str):
    """Extracts tar file. \n
        Args:
            - from_path (str): path where tar file is stored.
            - to_path (str): path where tar file will be extracted.
    """
    #Create path if it does not exist
    if not os.path.isdir(to_path):
        os.makedirs(to_path, exist_ok=True)

    #TAR file extracting process
    print("Extracting Data...")
    tar = tarfile.open(from_path)
    tar.extractall(to_path)
    tar.close()



#Create Images
def create_images(from_path: str, to_path: str, data_num: int, slice_num: int):
    """Extracts gz file. \n
        Args:
            - path (str): path where gz file is stored.
            - to_path (str): path where images will be extracted.
            - data_num (int): total image number.
            - slice_num (int): images number to be got from each patient.
    """
    data_counter = [0]

    #Path finding process
    for path1 in sorted(glob.glob(from_path + "/*")):
        for path2 in glob.glob(path1 + "/*.gz"):
            
            #Out direction of flair and seg
            out_dir = path2.split(".")[0]

            #Create out directions
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            #Extract corresponding data to output path
            ptl.extract_archive(path2, outdir=out_dir)
            os.remove(path2)

        create_nii_slices(path1, to_path, data_num, slice_num, data_counter)
        if data_counter[0] == data_num:
            break
        




#CREATE NII SLICES
def create_nii_slices(from_path: str, to_path: str, data_num: int, slice_num: int, data_counter):
    """Creates images from 3D image data. \n
        Args:
            - from_path (str): path where nii files is stored.
            - to_path (str): path where png files will be extracted.
            - data_num (int): total image number.
            - slice_num (int): images number to be got from each patient.
    """
    for path1_flair, path1_t1, path1_t1ce, path1_t2, path1_seg in zip(glob.glob(from_path + "/*_flair"), glob.glob(from_path + "/*_t1"), glob.glob(from_path + "/*_t1ce"), glob.glob(from_path + "/*_t2"), glob.glob(from_path + "/*_seg")):
        for path2_flair, path2_t1, path2_t1ce, path2_t2, path2_seg in zip(glob.glob(path1_flair + "/*_flair*"), glob.glob(path1_t1 + "/*_t1*"), glob.glob(path1_t1ce + "/*_t1ce*"), glob.glob(path1_t2 + "/*_t2*"), glob.glob(path1_seg + "/*_seg*")):
            #Path for output
            patient_num = from_path.split("/")[-1]

            #Flair
            flair_name = patient_num + "_flair"
            #T1
            t1_name = patient_num + "_t1"
            #T1ce
            t1ce_name = patient_num + "_t1ce"
            #T2
            t2_name = patient_num + "_t2"
            #Seg
            seg_name = patient_num + "_seg"


            #Flair
            if not os.path.exists(os.path.join(to_path, patient_num, flair_name)):
                os.makedirs(os.path.join(to_path, patient_num, flair_name))
            #T1
            if not os.path.exists(os.path.join(to_path, patient_num, t1_name)):
                os.makedirs(os.path.join(to_path, patient_num, t1_name))
            #T1ce
            if not os.path.exists(os.path.join(to_path, patient_num, t1ce_name)):
                os.makedirs(os.path.join(to_path, patient_num, t1ce_name))
            #T2
            if not os.path.exists(os.path.join(to_path, patient_num, t2_name)):
                os.makedirs(os.path.join(to_path, patient_num, t2_name))
            #Seg
            if not os.path.exists(os.path.join(to_path, patient_num, seg_name)):
                os.makedirs(os.path.join(to_path, patient_num, seg_name))

            
            #Flair
            flair = nib.load(path2_flair).get_fdata()
            #T1
            t1 = nib.load(path2_t1).get_fdata()
            #T1ce
            t1ce = nib.load(path2_t1ce).get_fdata()
            #T2
            t2 = nib.load(path2_t2).get_fdata()
            #Seg
            seg = nib.load(path2_seg).get_fdata()

            slice_counter = 0
            #Create image files
            for i in range(flair.shape[2]):
                if (seg[:, :, i] == 1).any() or (seg[:, :, i] == 2).any() or (seg[:, :, i] == 4).any():
                    
                    #Flair
                    flair_img = flair[:, :, i]
                    flair_img = np.uint8(flair_img / flair_img.max() * 255)
                    #T1
                    t1_img = t1[:, :, i]
                    t1_img = np.uint8(t1_img / t1_img.max() * 255)
                    #T1ce
                    t1ce_img = t1ce[:, :, i]
                    t1ce_img = np.uint8(t1ce_img / t1ce_img.max() * 255)
                    #T2
                    t2_img = t2[:, :, i]
                    t2_img = np.uint8(t2_img / t2_img.max() * 255)
                    #Seg
                    seg_img = seg[:, :, i]
                    seg_img = np.uint8(seg_img)

                    #Flair
                    flair_path = os.path.join(to_path, patient_num, flair_name, str(i) + ".png")
                    #T1
                    t1_path = os.path.join(to_path, patient_num, t1_name, str(i) + ".png")
                    #T1ce
                    t1ce_path = os.path.join(to_path, patient_num, t1ce_name, str(i) + ".png")
                    #T2
                    t2_path = os.path.join(to_path, patient_num, t2_name, str(i) + ".png")
                    #Seg
                    seg_path = os.path.join(to_path, patient_num, seg_name, str(i) + ".png")

                    #Flair
                    cv2.imwrite(flair_path, flair_img)
                    #T1
                    cv2.imwrite(t1_path, t1_img)
                    #T1ce
                    cv2.imwrite(t1ce_path, t1ce_img)
                    #T2
                    cv2.imwrite(t2_path, t2_img)
                    #Seg
                    cv2.imwrite(seg_path, seg_img)
                
                    data_counter[0] += 1
                    slice_counter += 1
                    if (slice_counter == slice_num) or (data_counter[0] == data_num):
                        break
            
            os.remove(path2_flair)
            os.remove(path2_t1)
            os.remove(path2_t1ce)
            os.remove(path2_t2)
            os.remove(path2_seg)