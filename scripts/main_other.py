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

### Need change later
def to_ply(point_clouds, colors):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    # out_colors = np.zeros(out_points.shape)
    print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
    verts = np.hstack([out_points, out_colors])

    mean = np.mean(verts[:, :3], axis=0)
    scaled_verts = verts[:, :3] - mean
    dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)

    verts = verts[indx]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    
    with open('res.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

### Need change later
def correspondences(img_points_1, img_points_2, img_points_3):
    cr_points_1 = []
    cr_points_2 = []

    for i in range(img_points_1.shape[0]):
        a = np.where(img_points_2 == img_points_1[i, :])
        if a[0].size != 0:
            cr_points_1.append(i)
            cr_points_2.append(a[0][0])

    mask_array_1 = np.ma.array(img_points_2, mask=False)
    mask_array_1.mask[cr_points_2] = True
    mask_array_1 = mask_array_1.compressed()
    mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

    mask_array_2 = np.ma.array(img_points_3, mask=False)
    mask_array_2.mask[cr_points_2] = True
    mask_array_2 = mask_array_2.compressed()
    mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)

    return np.array(cr_points_1), np.array(cr_points_2), mask_array_1, mask_array_2



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
    
    K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047 ],
                  [0, 2398.118540286656, 628.2649953288065],
                  [0, 0, 1]])

      
    poses = K.flatten()
    T1 = np.hstack((np.identity(3), np.array([[0], [0], [0]])))
    T2 = np.zeros(T1.shape)

    P1 = np.matmul(K, T1)

    points = np.zeros((1, 3))
    colors = np.zeros((1, 3))

    linking_matrix = find_features_to_linking_array(img1, img2)
    print("Created the Linking for the First Two Images Matrix")

    F, best_matches = FMatrix_RANSAC(linking_matrix.copy(),8,0.005)

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

    # ### That guy stuff
    # essential_matrix, em_mask = cv2.findEssentialMat(linking_matrix[:, 0:2], linking_matrix[:, 2:4], K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    # matches_1 = linking_matrix[:, 0:2][em_mask.ravel() == 1]
    # matches_2 = linking_matrix[:, 2:4][em_mask.ravel() == 1]

    # _, R2, C2, em_mask = cv2.recoverPose(essential_matrix, matches_1, matches_2, K)
    # matches_1 = matches_1[em_mask.ravel() > 0]
    # matches_2 = matches_2[em_mask.ravel() > 0]

    T2[:3, :3] = np.matmul(R2, T1[:3, :3])
    T2[:3, 3] = T1[:3, 3] + np.matmul(T1[:3, :3], C2.ravel())
    P2 = np.matmul(K, T2)

    poses = np.hstack((np.hstack((poses, P1.ravel())), P2.ravel()))

    print("Registered the First Two Images Together")

    for i in range(len(imgs) - 2):
        next_img = imgs[i+2]
        linking_matrix = find_features_to_linking_array(img2, next_img)

        
        # matches_2.T
        if i == 0:
            matches_1, matches_2, pts_3D = cv_triangulation(P1, P2, best_matches[:, 0:2], best_matches[:, 2:4])
            pts_3D = pts_3D.T[:, :3]
            # matches_2 = matches_2.T
            # _, _, _, inliers = cv2.solvePnPRansac(pts_3D, matches_2, K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
            # if inliers is not None:
            #     pts_3D = pts_3D[inliers[:, 0]]
            #     matches_2 = matches_2[inliers[:, 0]]

            corr_point1, corr_points_2, mask1, mask2 = correspondences(matches_2, linking_matrix[:, 0:2], linking_matrix[:, 2:4])
        else:
            matches_1, matches_2, pts_3D = cv_triangulation(P1, P2, matches_1, matches_2)
            pts_3D = cv2.convertPointsFromHomogeneous(pts_3D.T)
            pts_3D = pts_3D[:, 0, :]
            corr_point1, corr_points_2, mask1, mask2 = correspondences(matches_2.T, linking_matrix[:, 0:2], linking_matrix[:, 2:4])
        
        corr_points_3 = linking_matrix[:, 2:4][corr_points_2]
        corr_points_cur = linking_matrix[:, 0:2][corr_points_2]

        _, R, T, inliers = cv2.solvePnPRansac(pts_3D[corr_point1], corr_points_3, K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
        R = Rotation.from_rotvec(R.reshape((1,3))).as_matrix().reshape((3,3))

        print("Completed PnP Ransac for Image ",i+2)

        if inliers is not None:
            corr_points_3 = corr_points_3[inliers[:, 0]]
            pts_3D = pts_3D[inliers[:, 0]]
            corr_points_cur = corr_points_cur[inliers[:, 0]]

        T2 = np.hstack((R, T))
        P3 = np.matmul(K, T2)

        mask1, mask2, pts_3D = cv_triangulation(P2, P3, mask1, mask2)
        pts_3D = pts_3D.T[:, :3]

        poses = np.hstack((poses, P3.ravel()))

        points = np.vstack((points, pts_3D))
        points_left = np.array(mask2, dtype=np.int32)
        color_vector = np.array([next_img[l[1], l[0]] for l in points_left.T])
        colors = np.vstack((colors, color_vector))

        T1 = np.copy(T2)
        P1 = np.copy(P2)

        img1 = np.copy(img2)
        img2 = np.copy(next_img)
        matches_1 = np.copy(linking_matrix[:, 0:2])
        matches_2 = np.copy(linking_matrix[:, 2:4])
        P2 = np.copy(P3)

        print("Successfully Registered Image ",i+3)

    to_ply(points, colors)


if __name__ == "__main__":
    main("../data/images/monument/")