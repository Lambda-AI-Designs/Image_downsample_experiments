import cv2
import numpy as np
import matplotlib.pyplot as plt

#open initial test image
original = cv2.imread("test_image.jpg"); #import original 8k image
cv2.imshow("8K", original);

def manual_downsampling(original):
    rows, cols, channels = map(int, original.shape)
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                if (j %2 == 0):
                    original[i][j][k]=0;
    return original

#define manual downsampling functions
def downsample_cols(original):
    rows, cols, = map(int, original.shape)
    for i in range(cols):
        if(cols%2==0):
            original = np.delete(original, [i],axis=1)
    return original

def downsample_rows(original):
    rows, cols, = map(int, original.shape)
    for i in range(rows):
        if(rows%2==0):
            original = np.delete(original, [i],axis=0)
    return original

def downsample_complete(original):
    rows, cols, = map(int, original.shape)
    for i in range(rows):
        if (rows % 2 == 0):
            original = np.delete(original, [i], axis=0)
    for i in range(cols):
        if(cols%2==0):
            original = np.delete(original, [i],axis=1)
    return original

#Gaussian Pyramid Downsampling
rows, cols, channels = map(int, original.shape)
layer_1 = cv2.pyrDown(original, dstsize=(cols//2, rows//2))
cv2.imshow("layer 1", layer_1)
cv2.imwrite("gaussian.png",layer_1)
#manual downsampling
original0 = manual_downsampling(original)
cv2.imwrite("manual_color.png",original0)
#reopen grayscale image
original = cv2.imread("test_image.jpg",0)

original1 = downsample_cols(original)
cv2.imwrite("manual_columns.png",original1)
original2 = downsample_rows(original)
cv2.imwrite("manual_rows.png",original2)
original3 = downsample_complete(original)
cv2.imwrite("manual_rowscols.png",original3)
cv2.imshow("manual downsampling every other x1", original0)
cv2.imshow("manual downsampling columns", original1)
cv2.imshow("manual downsampling rows", original2)
cv2.imshow("manual downsampling complete", original3)
cv2.waitKey(0)
cv2.destroyAllWindows()