import torch
import numpy as np
from PIL import Image
from . import transforms
import matplotlib.pyplot as plt

def vis_patch(img, skg, texture_location, color='lab'):
    """
    将img在texture_location部分的像素贴到skg里面, 返回贴片后的skg图
    :param img: 原始图
    :param skg: 轮廓线条图
    :param texture_location: [x_center, y_center, patch_size]
    :param color: 颜色空间, 默认'lab', 这里应该跟主程序走, 设置什么的无所谓
    :return: 替换后的skg线条图.
    """
    batch_size, _, _, _ = img.size()
    if torch.cuda.is_available():
        img = img.cpu()
        skg = skg.cpu()

    img = img.numpy()
    skg = skg.numpy()

    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
        
    img_np = ToRGB(img)
    skg_np = ToRGB(skg)

    vis_skg = np.copy(skg_np)
    vis_img = np.copy(img_np)
    # plt.figure()
    # plt.subplot(131)
    # plt.title('vis_skg')
    # plt.imshow(vis_skg[0, :, :, :].transpose(1, 2, 0))  # numpy用transpose, pytorch用permute
    # plt.subplot(132)
    # plt.title('vis_img')
    # plt.imshow(vis_img[0, :, :, :].transpose(1, 2, 0))

    # print np.shape(vis_skg)
    for i in range(batch_size):
        for text_loc in texture_location[i]:
            xcenter, ycenter, size = text_loc
            xcenter = max(xcenter-int(size/2),0) + int(size/2)  # 保证不出边界
            ycenter = max(ycenter-int(size/2),0) + int(size/2)
            vis_skg[
                i, :,
                int(xcenter-size/2):int(xcenter+size/2),
                int(ycenter-size/2):int(ycenter+size/2)
            ] = vis_img[
                    i, :,
                    int(xcenter-size/2):int(xcenter+size/2),
                    int(ycenter-size/2):int(ycenter+size/2)
                ]
    # plt.subplot(133)
    # plt.title('vis_skg after')
    # plt.imshow(vis_skg[0, :, :, :].transpose(1, 2, 0))
    # plt.show()
    return vis_skg
    
def vis_image(img, color='lab'):
    if torch.cuda.is_available():
        img = img.cpu()

    img = img.numpy()

    if color == 'lab':
        ToRGB = transforms.toRGB()
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')

    # print np.shape(img)
    img_np = ToRGB(img)

    return img_np
