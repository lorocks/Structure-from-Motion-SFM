import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from camera_poses import *
from disparity import *
from epipolar_viz import *
from feature_extraction import *
from fmatrix_ransac import *
from fundamental_matrix import *
from pnp import *
from triangulation import *
from utils import *

def main():

    imgs = load_imgs("../data/images/")
    img1 = imgs[0]
    img2 = imgs[1]
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    
    print("Image 1 Shape: ",img1.shape)
    print("Image 2 Shape: ",img2.shape)
    
    kp1, features1 = detect_BRISK_features(copy.deepcopy(img1))
    kp2, features2 = detect_BRISK_features(copy.deepcopy(img2))
    # img1_pts and img2_pts: both [N,2] shape vectors
    # linking_matrix: [N,4] shape matrix
    img1_pts, img2_pts, linking_matrix = match_BRISK_features(kp1,features1,kp2,features2)

    # feature_matrix has a shape [N,M,2] --> N: total nnumber of images, M: total number of common features
    feature_matrix = construct_feature_matrix("../data/images/","1.jpg")
    print("Shape of the feature matrix: ",feature_matrix.shape)

    print("Matched BRISK Features")

    # F has shape [3,3]
    F, best_matches = FMatrix_RANSAC(linking_matrix.copy(),8,0.002)

    print("Calculated the Fundamental Matrix using RANSAC. Printing it below")
    print(F)

    # camera_poses: [[R1,C1], [R2,C2], [R3,C3], [R4,C4]] where Ri: [3,3] and Ci: [3,1]
    camera_poses = get_all_camera_poses(F,K,K)
    R1, C1 = np.eye(3), np.zeros((3,1))
    X_set = []
    for i, R_C in enumerate(camera_poses):
        R2, C2 = R_C
        X = linear_triangulation(img1_pts,img2_pts,C1,C2,R1,R2,K,K) # X: [N,4]
        X_set.append(X[:,:3])
    
    print("Finished Linear Triangulation")
    
    # X_set: [X1, X2, X3, X4] where Xi: [N,3]
    R2, C2, pts_3D = best_camera_pose(X_set,camera_poses) # pts_3D: [N,3]
    pts_3D = non_linear_triangulation(pts_3D,img1_pts,img2_pts,C1,C2,R1,R2,K,K) # pts_3D: [N,4] with last column of 1s

    print("Finished Non-Linear Triangulation")

if __name__ == "__main__":
    main()