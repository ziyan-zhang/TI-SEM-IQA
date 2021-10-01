"""-----------------------------------------------------
创建时间 :  2020/5/8  16:34
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import cv2

def qianjing_single(img_fpath, mask_fpath, out_fpath):
    img = cv2.imread(img_fpath, 0)
    mask = cv2.imread(mask_fpath, 0)
    mask_inverse = 1 - mask
    # img_qianjing = img*mask
    img_qianjing2 = cv2.bitwise_and(img, mask)
    img_qianjing2_lightbg = img_qianjing2 + mask_inverse*255
    cv2.imwrite(out_fpath, img_qianjing2_lightbg)

if __name__ == '__main__':
    img_fpath = 'E:\\SEMBig_Qianjing\\train_img\\01_001.tif'
    mask_fpath = 'E:\\SEMBig_Qianjing\\train_seg\\01_001binary.png'
    out_fpath = 'out.png'
    qianjing_single(img_fpath, mask_fpath, out_fpath)