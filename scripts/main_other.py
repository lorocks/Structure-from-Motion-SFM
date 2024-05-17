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
    
    #K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047 ],
    #              [0, 2398.118540286656, 628.2649953288065],
    #              [0, 0, 1]])
    
    K = np.array([[700.97, 0, 649.888],
                  [0, 701.302, 333.853],
                  [0, 0, 1]])
      
    poses = K.flatten()
    T1 = np.hstack((np.identity(3), np.array([[0], [0], [0]])))
    T2 = np.zeros(T1.shape)

    P1 = np.matmul(K, T1)

    points = np.zeros((1, 3))
    colors = np.zeros((1, 3))

    linking_matrix = find_features_to_linking_array(img1, img2)
    print("Created the Linking for the First Two Images Matrix")

    F, best_matches = FMatrix_RANSAC(linking_matrix.copy(),8,0.5)

    print("Calculated the Fundamental Matrix for the First Two Images using RANSAC. Printing it below")
    print(F)

    # camera_poses: [[R1,C1], [R2,C2], [R3,C3], [R4,C4]] where Ri: [3,3] and Ci: [3,1]
    camera_poses = get_all_camera_poses(F,K,K)
    R1, C1 = np.eye(3), np.zeros((3))
    X_set = []
    for i, R_C in enumerate(camera_poses):
        R2, C2 = R_C
        X = linear_triangulation(best_matches[:, 0:2],best_matches[:, 2:4],C1,C2,R1,R2,K,K) # X: [N,4]
        X_set.append(X[:,:3])
    
    print("Finished Initial Linear Triangulation")
    
    # X_set: [X1, X2, X3, X4] where Xi: [N,3]
    R2, C2, pts_3D = best_camera_pose(X_set,camera_poses) # pts_3D: [N,3]

    T2[:3, :3] = np.matmul(R2, T1[:3, :3])
    T2[:3, 3] = T1[:3, 3] + np.matmul(T1[:3, :3], C2.ravel())
    P2 = np.matmul(K, T2)

    poses = np.hstack((np.hstack((poses, P1.ravel())), P2.ravel()))

    print("Registered the First Two Images Together")

    matches_1 = best_matches[:, 0:2]
    matches_2 = best_matches[:, 2:4]

    for i in range(len(imgs) - 2):
        next_img = imgs[i+2]
        linking_matrix = find_features_to_linking_array(img2, next_img, 0.7)

        # CAn use the custom triangulation fn if able to disambiguate the projection into K, R and T
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

        # CAn use the custom triangulation fn if able to disambiguate the projection into K, R and T
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

    save_ply(points, colors, f'../data/output/res_{img_dir.split("/")[-2]}.ply')


if __name__ == "__main__":
    main("../data/images/turtle/")