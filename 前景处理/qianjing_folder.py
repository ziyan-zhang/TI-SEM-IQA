"""-----------------------------------------------------
创建时间 :  2020/5/8  16:38
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
from qianiing_single import qianjing_single
from tqdm import tqdm


img_folder = 'E:\\SEMBig_lightBG\\val_img'
mask_folder = 'E:\\SEMBig_lightBG\\val_seg'

imgNames = os.listdir(img_folder)
maskNames = os.listdir(mask_folder)

length = len(imgNames)
out_dir = 'E:\\SEMBig_lightBG\\light'

for i in tqdm(range(length)):
    imgName = imgNames[i]
    maskName = maskNames[i]
    img_fpath = os.path.join(img_folder, imgName)
    mask_fpath = os.path.join(mask_folder, maskName)
    out_fpath = os.path.join(out_dir, imgName)

    qianjing_single(img_fpath, mask_fpath, out_fpath)