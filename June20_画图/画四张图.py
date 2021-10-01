from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


W = 3231
H = 3569

img_youshang= Image.open('C:\\Users\\ZhangZiyan\\Desktop\\右上.jpg')
img_youzuo = Image.open('C:\\Users\\ZhangZiyan\\Desktop\\右左.jpg')
img_xiazuo = Image.open('C:\\Users\\ZhangZiyan\\Desktop\\下左.jpg')
img_xiashang = Image.open('C:\\Users\\ZhangZiyan\\Desktop\\下上.jpg')


aa = np.zeros((2*W+10, 2*H+10))

aa[0:H, 0:W] = img_youshang.convert('L')
aa[0:H, W+10:2*W+10] = img_youzuo.convert('L')
aa[H+10:2*H+10, 0:W] = img_xiazuo.convert('L')
aa[H+10:2*H+10, W+10:2*W+10] = img_xiashang.convert('L')
plt.imshow(aa)
