import numpy as np
import cv2 
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# Mohammad H. Mansoor

def luv_space_gamma(src, gamma):
    luv = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)
    #// extract luminance channel
    l = luv[:,:,0]
    #// normalize 
    l = l / 255.0
    #// apply power transform
    l = np.power(l, gamma)
    #// scale back
    l = l * 255
    luv[:,:,0] = l.astype(np.uint8)
    rgb = cv2.cvtColor(luv, cv2.COLOR_LUV2BGR)
    return rgb

def skin_rgb_threshold(src):
    # extract color channels and save as SIGNED ints
    # need the extra width to do subraction
    b = src[:,:,0].astype(np.int16)
    g = src[:,:,1].astype(np.int16)
    r = src[:,:,2].astype(np.int16)

    skin_mask =                                    \
          (r > 96) & (g > 40) & (b > 10)           \
        & ((src.max() - src.min()) > 15)           \
        & (np.abs(r-g) > 15) & (r > g) & (r > b)    

    return src * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)

def find_local_min( hist ):

    kern = np.array(
            [2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             -3,-3,-3,-3
             -3,-3,-3,-3
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2])
    #// theres a lot of 0's in there what will throw off 
    #// the convolution
    hist[0] = 0
    deriv = np.convolve(hist, kern, mode='same')
    threshold = deriv.argmax()
    return threshold, deriv

im1 = cv2.imread('face_good.bmp')
plt.title("Before Skin Detection")
img = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)

img = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
skin_detect = skin_rgb_threshold(im1)
plt.figure()
plt.title("After Skin Detection")
plt.imshow(cv2.cvtColor(skin_detect, cv2.COLOR_BGR2RGB))

im2 = cv2.imread('face_dark.bmp')
plt.figure()
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
plt.title("Before Skin Detection")
plt.imshow(img2,cmap='gray', vmin=0, vmax=255)

gamma = luv_space_gamma(im2, 0.6)
plt.figure()
plt.title('rgb')
plt.imshow(cv2.cvtColor(im2,cv2.COLOR_BGR2RGB))

plt.figure()
plt.title('RGB with LUV')
plt.imshow(cv2.cvtColor(gamma, cv2.COLOR_BGR2RGB))

luv = cv2.cvtColor(im2, cv2.COLOR_BGR2LUV)
rgb = cv2.cvtColor(luv, cv2.COLOR_LUV2BGR)
skin_detect2 = skin_rgb_threshold(rgb)
plt.figure()
plt.title("Low Light After Skin Detect")
plt.imshow(cv2.cvtColor(skin_detect2, cv2.COLOR_BGR2RGB))

luv = cv2.cvtColor(im2, cv2.COLOR_BGR2LUV)
l = luv[:,:,0]
u = luv[:,:,1]
v = luv[:,:,2]

histogram, bins = np.histogram(l.ravel(), 256, [0, 256])
thresh, derivative = find_local_min(histogram)

l = (l < thresh) * l
luv[:,:,0] = l

rgb = cv2.cvtColor(luv, cv2.COLOR_LUV2BGR)
skin_detect3 = skin_rgb_threshold(rgb)
plt.figure()
plt.title("Background Removed")
plt.imshow(cv2.cvtColor(skin_detect3, cv2.COLOR_BGR2RGB))

plt.show()