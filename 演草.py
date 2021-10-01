import matplotlib.pyplot as plt


plt.figure()
plt.subplot(121)
plt.imshow(img_read_i[0, :, :])
plt.title('没有norm')
plt.subplot(122)
plt.imshow(img_read_i_norm[0, :, :])
plt.title('norm了')
plt.show()

