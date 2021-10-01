import cv2
import matplotlib.pyplot as plt
from 画图函数 import huaquan, caijian, maskit, tihuan
import os

outDir = 'xiaotu3xiao_quanxin/'
if not os.path.exists(outDir):
    os.mkdir(outDir)

img = cv2.imread('sk_img.png', 1)
skg = cv2.imread('sk_xi.png', 1)

# img = img[:, :884, :]
# skg = skg[:, :884, :]

img_ori = img.copy()

ST1 = 62
ST2 = 80

# img = cv2.imread('image3xiao.png', 1)

zhongjiankuai = caijian(img, ST2)
cv2.imwrite(outDir+'zhongjiankuai.png', zhongjiankuai)


img_greened = huaquan(img, ST1)
img_magentaed = huaquan(img, st=ST2, the_color=[225, 0, 225])  # magenta

skg_greened = huaquan(skg, ST1)
skg_magentaed = huaquan(skg, st=ST2, the_color=[225, 0, 225])  # magenta

img_cut_green = caijian(img_magentaed, ST1)
img_cut_magenta = caijian(img_magentaed, ST2)

skg_center = caijian(skg, ST2)

masked = maskit(img, ST2)

synthesised_img = tihuan(img, skg, ST2)

cv2.imwrite(outDir+'img.png', img_magentaed)
cv2.imwrite(outDir+'skg.png', skg_magentaed)

cv2.imwrite(outDir+'groundtruth.png', img_cut_green)
cv2.imwrite(outDir+'masked.png', masked)
cv2.imwrite(outDir+'skg_center.png', skg_center)
cv2.imwrite(outDir+'synthesised_img.png', synthesised_img)

plt.subplot(331)
plt.imshow(img_ori)
plt.title(str(img_ori.shape))
plt.subplot(332)
plt.imshow(img_greened)
plt.title(str(img_greened.shape))
plt.subplot(333)
plt.imshow(img_magentaed)
plt.title(str(img_magentaed.shape))
plt.subplot(334)
plt.imshow(img_cut_green)
plt.title(str(img_cut_green.shape))
plt.subplot(335)
plt.imshow(img_cut_magenta)
plt.title(str(img_cut_magenta.shape))
plt.subplot(336)
plt.imshow(masked)
plt.title('masked.png')
plt.subplot(337)
plt.imshow(skg_center)
plt.title('skgcenter' + str(skg_center.shape))
plt.subplot(338)
plt.imshow(synthesised_img)
plt.title('synthesised_img' +str(synthesised_img.shape))
plt.subplot(339)
plt.imshow(skg_magentaed)
plt.title('skg_magentaed' +str(skg_magentaed.shape))