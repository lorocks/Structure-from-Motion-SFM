import numpy as np
import cv2

# Perform Bundle Adjustment to Improve the Estimate of the Reconstructed 3D point locations
def bundle_adjustment(img_idx,X,feature_matrix,all_RC,K):
    pass # Returns X of shape [N,4], with N: Numnber of points=feature_matrix.shape[1], 4'th column (index 3) of X is just 1s 