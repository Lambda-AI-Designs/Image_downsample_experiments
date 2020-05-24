import cv2
import numpy as np
import time
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
    original = np.delete(original, range(0,cols,2),axis=1)
    return original

def downsample_rows(original):
    rows, cols, = map(int, original.shape)
    original = np.delete(original, range(0, rows, 2), axis=0)
    return original

def downsample_complete(original):
    rows, cols, = map(int, original.shape)
    original = np.delete(original, range(0, cols, 2), axis=1)
    original = np.delete(original, range(0, rows, 2), axis=0)
    return original

#Gaussian Pyramid Downsampling
rows, cols, channels = map(int, original.shape)
start = time.time()
layer_1 = cv2.pyrDown(original, dstsize=(cols//2, rows//2))
print("Gaussian took %f seconds"%(time.time()-start))
cv2.imshow("layer 1", layer_1)
cv2.imwrite("gaussian.png",layer_1)
#manual downsampling
start = time.time()
original0 = manual_downsampling(original)
print("Manual color took %f seconds"%(time.time()-start))
cv2.imwrite("manual_color.png",original0)
#reopen grayscale image
original = cv2.imread("test_image.jpg",0)

start = time.time()
original1 = downsample_cols(original)
print("Manual columns took %f seconds"%(time.time()-start))
cv2.imwrite("manual_columns.png",original1)

start = time.time()
original2 = downsample_rows(original)
print("Manual rows took %f seconds"%(time.time()-start))
cv2.imwrite("manual_rows.png",original2)

start = time.time()
original3 = downsample_complete(original)
print("Manual rows+columns took %f seconds"%(time.time()-start))
cv2.imwrite("manual_rowscols.png",original3)

#Classic resize method
start = time.time()
scale_percent = 50 # percent of original size
width = int(original.shape[1] * scale_percent / 100)
height = int(original.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
print("Classic resize took %f seconds"%(time.time()-start))

cv2.imshow("manual downsampling every other x1", original0)
cv2.imshow("manual downsampling columns", original1)
cv2.imshow("manual downsampling rows", original2)
cv2.imshow("manual downsampling complete", original3)
cv2.waitKey(0)
cv2.destroyAllWindows()