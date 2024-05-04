import numpy as np
import cv2
import tqdm

# Function to calculate SSD and then fill a disparity array
def disparitySSD(image_left, image_right):
    gray1 = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    window_size = 30
    block = 7
    disparity = np.zeros(gray1.shape)
    # Calculate ssd between images
    # Compares a block in image 1 within a window range in image 2
    for i in tqdm(range(block, gray1.shape[0] - block - 1)):
        for j in range(block + window_size, gray1.shape[1] - block - 1):
            ssd = np.empty([window_size,1])
            l = gray1[(i-block):(i+block), (j-block):(j+block)]
            for f in range(0,window_size):
                r = gray2[(i-block):(i+block), (j-f-block):(j-f+block)]
                ssd[f] = np.sum((l[:,:] - r[:,:])**2)
            disparity[i,j] = np.argmin(ssd)
    return disparity

# Function to calculate SAD and then fill a disparity array
def disparitySAD(image_left,image_right):
    block = 3
    local_window = 5
    left = np.asarray(image_left).astype(int)
    right = np.asarray(image_right).astype(int)
    height, width, _ = left.shape
    disparity = np.zeros((height, width))
    #going over each pixel
    for y in tqdm(range(block, height-block)):
        for x in range(block, width-block):
            left_local = left[y:y + block, x:x + block]
            x_min = max(0, x - local_window)
            x_max = min(right.shape[1], x + local_window)
            min_sad = None
            index_min = None
            first = True
            for x in range(x_min, x_max):
                right_local = right[y: y+block,x: x+block]
                if left_local.shape == right_local.shape:
                    sad = np.sum(abs(left_local - right_local))
                else:
                    sad = -1
                if first:
                    min_sad = sad
                    index_min = (y, x)
                    first = False
                else:
                    if sad < min_sad:
                        min_sad = sad
                        index_min = (y, x)
            disparity[y, x] = abs(index_min[1] - x)
    return disparity