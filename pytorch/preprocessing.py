import os
import glob

import tarfile
import patoolib as ptl

import numpy as np
import cv2
import nibabel as nib



#EXTRACT TAR FILE
def extract_tar_file(from_path, to_path):
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



#EXTRACT GZ FILE
def extract_gz_file(path):
    """Extracts gz file. \n
        Args:
            - path (str): path where gz file is stored.
    """
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




#CREATE NII SLICES
def create_nii_slices(from_path, to_path):
    """Creates images from 3D image data. \n
        Args:
            - from_path (str): path where nii files is stored.
            - to_path (str): path where png files will be extracted.
    """
    for path1 in glob.glob(from_path + "/*"):
        for path2_f, path2_s in zip(glob.glob(path1 + "/*_flair"), glob.glob(path1 + "/*_seg")):
            for path3_f, path3_s in zip(glob.glob(path2_f + "/*_flair*"), glob.glob(path2_s + "/*_seg*")):
                #Path for output
                patient_num = path1.split("/")[-1]
                flair_name = patient_num + "_flair"
                seg_name = patient_num + "_seg"

                #Create output paths if it does not exist
                if not os.path.exists(os.path.join(to_path, patient_num, flair_name)):
                    os.makedirs(os.path.join(to_path, patient_num, flair_name))
                    os.makedirs(os.path.join(to_path, patient_num, seg_name))
                
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
                        flair_path = os.path.join(to_path, patient_num, flair_name, str(i) + ".png")
                        seg_path = os.path.join(to_path, patient_num, seg_name, str(i) + ".png")

                        #Create images
                        cv2.imwrite(flair_path, flair_img)
                        cv2.imwrite(seg_path, seg_img)