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
from save_3D import *
from utils import *


# Main Implementation
def main(img_dir):
    imgs = load_imgs(img_dir)

    img1 = imgs[0]
    img2 = imgs[1]

    print("Image 1 Shape: ",img1.shape)
    print("Image 2 Shape: ",img2.shape)

    # K = np.array([[1401.9, 0, 340.444],
    #               [0, 1402, 174.427],
    #               [0, 0, 1]])

    k1=-0.16762533591578163
    k2=0.015365688221134153
    k3=0.006560209455351331
    p1=0.0014424610109027888
    p2=9.519472036645519e-05

    # K = np.array([[700.45, 0, 649.385],
    #                 [0, 700.45, 339.603],
    #                 [0, 0, 1]])
    

    # This one works best for now - turtle
    K = np.array([[1401.9, 0, 649.888],
                  [0, 1402, 333.853],
                  [0, 0, 1]])

    distortion = np.array([k1, k2, k3, p1, p2])

    w = 1280
    h = 720

    K, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w,h), 1, (w,h))
    
    # This one works best for now - monument
    K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047 ],
                  [0, 2398.118540286656, 628.2649953288065],
                  [0, 0, 1]])

      
    T1 = np.hstack((np.identity(3), np.array([[0], [0], [0]])))
    T2 = np.zeros(T1.shape)

    P1 = np.matmul(K, T1)

    points = np.zeros((1, 3))
    colors = np.zeros((1, 3))

    linking_matrix = find_features_to_linking_array(img1, img2, 0.7)
    print("Created the Linking for the First Two Images Matrix")

    essential_matrix, em_mask = cv2.findEssentialMat(linking_matrix[:, 0:2], linking_matrix[:, 2:4], K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    matches_1 = linking_matrix[:, 0:2][em_mask.ravel() == 1]
    matches_2 = linking_matrix[:, 2:4][em_mask.ravel() == 1]

    _, R2, C2, em_mask = cv2.recoverPose(essential_matrix, matches_1, matches_2, K)
    matches_1_temp = matches_1[em_mask.ravel() > 0]
    matches_2_temp = matches_2[em_mask.ravel() > 0]

    if matches_1_temp.shape[0] != 0:
        matches_1 = matches_1_temp
        matches_2 = matches_2_temp

    T2[:3, :3] = np.matmul(R2, T1[:3, :3])
    T2[:3, 3] = T1[:3, 3] + np.matmul(T1[:3, :3], C2.ravel())
    P2 = np.matmul(K, T2)

    print("Registered the First Two Images Together")

    for i in range(len(imgs) - 2):
        next_img = imgs[i+2]
        linking_matrix = find_features_to_linking_array(img2, next_img, 0.7)

        pts_3D = cv2.triangulatePoints(P1, P2, matches_1.T, matches_2.T)
        pts_3D = pts_3D / pts_3D[3]

        if i == 0:
            pts_3D = pts_3D.T[:, :3]
            _, _, _, inliers = cv2.solvePnPRansac(pts_3D, matches_2, K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
            if inliers is not None:
                pts_3D = pts_3D[inliers[:, 0]]
                matches_2 = matches_2[inliers[:, 0]]

        else:
            pts_3D = cv2.convertPointsFromHomogeneous(pts_3D.T)
            pts_3D = pts_3D[:, 0, :]
            

        corr_points1 = []
        corr_points2 = []

        for j in range(matches_2.shape[0]):
            if np.where(linking_matrix[:, 0:2] == matches_2[j, :])[0].size != 0:
                corr_points1.append(j)
                corr_points2.append(np.where(linking_matrix[:, 0:2] == matches_2[j, :])[0][0])
        
        mask1 = np.ma.array(linking_matrix[:, 0:2], mask=False)
        mask1.mask[corr_points2] = True
        mask1 = mask1.compressed()
        mask1 = mask1.reshape(int(mask1.shape[0] / 2), 2)

        mask2 = np.ma.array(linking_matrix[:, 2:4], mask=False)
        mask2.mask[corr_points2] = True
        mask2 = mask2.compressed()
        mask2 = mask2.reshape(int(mask2.shape[0] / 2), 2)

        _, R, T, inliers = cv2.solvePnPRansac(pts_3D[corr_points1], linking_matrix[:, 2:4][corr_points2], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
        R = Rotation.from_rotvec(R.reshape((1,3))).as_matrix().reshape((3,3))

        print("Completed PnP Ransac for Image ",i+2)

        if inliers is not None:
            pts_3D = pts_3D[inliers[:, 0]]

        T2 = np.hstack((R, T))
        P3 = np.matmul(K, T2)

        pts_3D = cv2.triangulatePoints(P2, P3, mask1.T, mask2.T)
        pts_3D = pts_3D / pts_3D[3]

        pts_3D = pts_3D.T[:, :3]
        points = np.vstack((points, pts_3D))
        points_left = np.array(mask2.T, dtype=np.int32)
        color_vector = np.array([next_img[l[1], l[0]] for l in points_left.T])
        colors = np.vstack((colors, color_vector))

        img1 = np.copy(img2)
        img2 = np.copy(next_img)
        matches_1 = np.copy(linking_matrix[:, 0:2])
        matches_2 = np.copy(linking_matrix[:, 2:4])
        T1 = np.copy(T2)
        P1 = np.copy(P2)
        P2 = np.copy(P3)

        print("Successfully Registered Image ",i+3)

    save_ply(points, colors, f'../data/output/cv_res_{img_dir.split("/")[-2]}.ply')


if __name__ == "__main__":
    main("../data/images/turtle/")