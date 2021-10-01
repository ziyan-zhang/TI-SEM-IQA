import cv2
TK = 3
def huaquan(img, st=None, tk=TK, the_color=None):
    the_green = [0, 255, 0]
    if the_color == None:
        the_color=the_green
    H, W = img.shape[:2]
    if st == None:
        st = round(H/4)

    img_ori = img.copy()
    img[st: st+tk, st:W-st, :] = the_color
    img[H-st-tk: H-st, st:W-st, :] = the_color
    img[st: H-st, st: st+tk, :] = the_color
    img[st: H-st, W-st-tk: W-st, :] = the_color

    return img

def huaquan0(img, st=None, tk=TK, the_color=None):
    return img

def caijian(img, st=None):
    H, W = img.shape[:2]
    if st == None:
        st = round(H/4)

    img = img[st: H-st, st:W-st, :]

    return img

def maskit(img, st=None, tk=TK):
    H, W = img.shape[:2]
    if st == None:
        st = round(H/4)

    img_ori = img.copy()
    img_ori[st+tk: H-st-tk, st+tk:W-st-tk, :] = [255, 255, 255]

    return img_ori

def tihuan(img, skg, st=None, tk=TK):
    H, W = img.shape[:2]
    if st == None:
        st = round(H/4)

    img_ori = img.copy()
    img_ori[st+tk: H-st-tk, st+tk:W-st-tk, :] = skg[st+tk: H-st-tk, st+tk:W-st-tk, :]

    return img_ori


if __name__ == '__main__':
    img = cv2.imread('01_001.tif', 1)
    img = img[:, :884, :]

    huaquan(img)

