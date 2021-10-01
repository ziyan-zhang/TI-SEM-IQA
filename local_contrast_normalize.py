from __future__ import division
import torch
from torch import Tensor
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from scipy.ndimage.filters import convolve
import torchvision
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus'] = False

def local_cst_norm(tensor, KernelSize = 3, Cval = 0.0):
    C, H, W = tensor.size()
    kernel = np.ones((KernelSize, KernelSize)) / (KernelSize**2)

    arr = np.array(tensor)
    local_avg_arr = np.array([convolve(arr[c], kernel, mode='constant', cval=Cval)
                              for c in range(C)]) # An array that has shape(C, H, W)
                                                  # Each element [c, h, w] is the average of the values
                                                  # in the window that has arr[c,h,w] at the center.

    arr_square = np.square(arr)
    local_avg_arr_square = np.array([convolve(arr_square[c], kernel, mode='constant', cval=Cval)
                              for c in range(C)]) # An array that has shape(C, H, W)
                                                  # Each element [c, h, w] is the average of the values
                                                  # in the window that has arr_square[c,h,w] at the center.
    local_sum_arr_square = local_avg_arr_square
    local_norm_arr = np.sqrt(local_sum_arr_square) # The tensor of local Euclidean norms.

    local_avg_divided_by_norm = local_avg_arr / (1e-8 + local_norm_arr)
    result_arr = np.minimum(local_avg_arr, local_avg_divided_by_norm)

    return torch.Tensor(local_avg_divided_by_norm)


if __name__ == '__main__':
    im = Image.open('2017-12-16-09-47-27-1200x800.jpg')
    # %%
    tensor = torchvision.transforms.ToTensor()(im)
    # %%
    lcned_tensor = local_cst_norm(tensor)
    # %%
    newpil = torchvision.transforms.ToPILImage()(lcned_tensor)

    plt.subplot(121)
    plt.imshow(im)
    plt.title('原图')
    plt.subplot(122)
    plt.imshow(lcned_tensor.permute(1,2,0))
    plt.title('新图')
    plt.show()