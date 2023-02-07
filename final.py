import numpy as np
import cv2

# image = cv2.imread('Classification Dataset/BPL-Ultima-PrimeD-A/prashant_icu_mon--1_2022_11_26_23_15_1.jpeg')
# importing image
image = cv2.imread('graph/Screenshot from 2023-02-07 07-19-34.png')
# cv2.waitKey(0)
# sharpening image and then converting to grayscale
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
kernel = kernel*1.2
image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

gray = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur (3, 3)
gray = cv2.GaussianBlur(gray, (3,3), 0)

# Applying THresholdinf through maxval
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
threshold = (maxVal)*0.7
assignvalue = 255 # Value to assign the pixel if the threshold is met
threshold_method = cv2.THRESH_BINARY

_, thres_img = cv2.threshold(gray,threshold,assignvalue,threshold_method)

# Sharpening image
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
kernel = kernel*1.2
image_sharp = cv2.filter2D(src=thres_img, ddepth=-1, kernel=kernel)

# edge detection
kernel_edge = np.array([[-1, -1, -1],
                        [-1, 10, -1],
                        [-1, -1, -1]])
edge_img = cv2.filter2D(src=thres_img, ddepth=-1, kernel=kernel_edge)

# function to take avg
def avg_final(im):
    im = im
    rows = len(im)
    cols = len(im[0])
    y_max = rows
    y_min = 0
    y = []
    for i in range(cols):
        for j in range(rows-1, 0, -1):
            if im[j][i]:
                sum=j
                count=1
                im[j][i] = 0
                for k in range(j-1, j-6, -1):
                    if im[k][i]:
                        sum+=k
                        count+=1
                        im[k][i]=0
                avg = int(sum/count)
                im[avg][i]=255

                y.append(rows-avg)
                if avg>y_max: y_max=avg
                if avg<y_min: y_min=avg

                j=j-5
                for m in range(j, -1, -1):
                    im[m][i]=0
            else:
                y.append(None)

    y = np.array(y)

    return (im, y_max, y_min, y)

final_plot, y_max, y_min, y = avg_final(edge_img)

# to create disconnected plots of final plot
# def plots(im, y_diff, y):
#     lst = []
    
#     for i in range(1, len(y)):
#         if y[i]:
#             y_prev=y[i]
    
#     for 
            

cv2.imshow('sharp1', image)
cv2.waitKey(0)

# plotting on matplotlib
import matplotlib.pyplot as plt
x = [i for i in range(0, len(y)) if y[i] != None]
y = [y[j] for j in range(0, len(y)) if y[j] != None]
plt.plot(x, y)
plt.show()

# thres_sharp = cv2.filter2D(src=thres_img, ddepth=-1, kernel=kernel)
cv2.imshow('sharp', final_plot)
