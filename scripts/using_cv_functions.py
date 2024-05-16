import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from scipy.spatial.transform import Rotation
from bundle_adjustment import *
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
    num_imgs = len(imgs)
    img1 = imgs[0]
    img2 = imgs[1]
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    
    print("Image 1 Shape: ",img1.shape)
    print("Image 2 Shape: ",img2.shape)
    
    '''kp1, features1 = detect_BRISK_features(copy.deepcopy(img1))
    kp2, features2 = detect_BRISK_features(copy.deepcopy(img2))
    # img1_pts and img2_pts: both [N,2] shape vectors
    # linking_matrix: [N,4] shape matrix
    img1_pts, img2_pts, linking_matrix = match_BRISK_features(kp1,features1,kp2,features2)
    print("Matched BRISK Features")'''

    # feature_matrix has a shape [N,M,2] --> N: total nnumber of images, M: total number of common features
    feature_matrix = construct_feature_matrix("../data/images/","1.jpg")
    print("Calculated the Feature Matrix")
    print("Shape of the Feature Matrix: ",feature_matrix.shape)

    img1_pts = feature_matrix[0,:,:].copy().reshape((-1,2)).astype(np.int32)
    img2_pts = feature_matrix[1,:,:].copy().reshape((-1,2)).astype(np.int32)
    linking_matrix = create_linking_matrix(img1_pts,img2_pts)
    print("Created the Linking for the First Two Images Matrix")

    # F has shape [3,3]
    F, mask = cv2.findFundamentalMat(img1_pts,img2_pts)

    print("Calculated the Fundamental Matrix for the First Two Images using RANSAC. Printing it below")
    print(F)

    # camera_poses: [[R1,C1], [R2,C2], [R3,C3], [R4,C4]] where Ri: [3,3] and Ci: [3,1]
    camera_poses = get_all_camera_poses(F,K,K)
    R1, C1 = np.eye(3), np.zeros((3))
    X_set = []
    for i, R_C in enumerate(camera_poses):
        R2, C2 = R_C
        P1, P2 = projection_matrix(K,R1,C1), projection_matrix(K,R2,C2)
        triangle_points = cv2.triangulatePoints(P1, P2, np.float32(img1_pts.reshape((2,-1))), np.float32(img2_pts.reshape((2,-1))))
        X_set.append(np.array(triangle_points)[0:3,:].reshape((-1,3)))
    
    print("Finished Initial Linear Triangulation")
    
    # X_set: [X1, X2, X3, X4] where Xi: [N,3]
    R2, C2, pts_3D = best_camera_pose(X_set,camera_poses) # pts_3D: [N,3]
    #pts_3D = non_linear_triangulation(pts_3D,img1_pts,img2_pts,C1,C2,R1,R2,K,K) # pts_3D: [N,4] with last column of 1s

    print("Finished Initial Non-Linear Triangulation")

    print("Registered the First Two Images Together")

    all_RC = [[R1,C1], [R2,C2]]
    # print(pts_3D[:, :3])

    for i in range(2,num_imgs):
        img_pts = feature_matrix[i,:,:].copy().reshape((-1,2)).astype(np.float32)
        print(np.array(pts_3D[:, :3]).shape, np.array(img_pts).shape)
        _, R_i, C_i = cv2.solvePnP(pts_3D[:,:3],img_pts,K,np.array([0,0,0,0]))
        R_i = Rotation.from_rotvec(R_i.reshape((1,3))).as_matrix().reshape((3,3))
        C_i = C_i.reshape((3,1))
        print("Completed Non-Linear PnP for Image ",i)
        all_RC.append([R_i,C_i])
        for j in range(i):
            img_ref_pts = feature_matrix[j,:,:].copy().reshape((-1,2)).astype(np.int32)
            R1, C1, R2, C2 = all_RC[j][0], all_RC[j][1], all_RC[i][0], all_RC[i][1]
            P1, P2 = projection_matrix(K,R1,C1), projection_matrix(K,R2,C2)
            triangle_points = cv2.triangulatePoints(P1, P2, np.float32(img_ref_pts.reshape((2,-1))), np.float32(img_pts.reshape((2,-1))))
            X = np.array(triangle_points)[0:3,:].reshape((-1,3))
            pts_3D = X
        # all_RC, pts_3D = bundle_adjustment(i,X,feature_matrix,all_RC,K)
        print("Successfully Registered Image ",i+1)
        # print(pts_3D[:, :3])
    print("Finished Registering all the Images")
    
    print("Ploting thw 3D Point-Cloud in a new Open3D Window")
    plot_3D_points(pts_3D)

if __name__ == "__main__":
    main()