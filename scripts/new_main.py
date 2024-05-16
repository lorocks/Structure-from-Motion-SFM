import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
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

def to_ply(point_clouds):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = np.zeros(out_points.shape)
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


def main():
    imgs = load_imgs("../data/images/")
    num_imgs = len(imgs)
    img1 = imgs[0]
    img2 = imgs[1]

    K = np.array([[350.485, 0, 340.444],
                  [0, 350.651, 174.427],
                  [0, 0, 1]])
    
    print("Image 1 Shape: ",img1.shape)
    print("Image 2 Shape: ",img2.shape)

    pose_array = K.ravel()

    T1 = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0]
        ])

    pose_1 = np.matmul(K, T1)
    # pose_2 = np.empty((3, 4)) 
    points_3D = np.zeros((1, 3))

    linking_matrix = find_features_to_linking_array(img1, img2)

    # F has shape [3,3]
    F, best_matches = FMatrix_RANSAC(linking_matrix.copy(),8,0.002)

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
    pts_3D = non_linear_triangulation(pts_3D,best_matches[:, 0:2],best_matches[:, 2:4],C1,C2,R1,R2,K,K) # pts_3D: [N,4] with last column of 1s

    T2 = np.empty((3, 4))
    T2[:3, :3] = np.matmul(R2, T1[:3, :3])
    T2[:3, 3] = T1[:3, 3] + np.matmul(T1[:3, :3], C2.ravel())

    pose_2 = np.matmul(K, T2)


    pose_array = np.hstack((np.hstack((pose_array, pose_1.ravel())), pose_2.ravel()))

    # for i in range(len(imgs) - 2):
    for i in range(2):
        next_img = imgs[i+2]
        linking_matrix = find_features_to_linking_array(img2, next_img)

        R_i, C_i = pnp_ransac(pts_3D,linking_matrix[:, 2:4],K)
        print("Completed PnP Ransac for Image ",i)
        R_i, C_i = non_linear_pnp(pts_3D,linking_matrix[:, 2:4],K,R_i,C_i)
        print("Completed Non-Linear PnP for Image ",i)

        T2 = np.hstack((R_i, C_i))
        pose_3 = np.matmul(K, T2)

        X = linear_triangulation(linking_matrix[:, 0:2],linking_matrix[:, 2:4],C1,C2,R1,R2,K,K)
        X = non_linear_triangulation(X,linking_matrix[:, 0:2],linking_matrix[:, 2:4],C1,C2,R1,R2,K,K)

        pose_array = np.hstack((pose_array, pose_3.ravel()))

        points_3D = np.vstack((points_3D, X[:, 0, :]))

        T1 = np.copy(T2)
        pose_1 = np.copy(pose_2)
        img1 = np.copy(img2)
        img2 = np.copy(next_img)
        pose_1 = np.copy(pose_2)

    to_ply(points_3D)

# that riangulation fn kinda sussy

if __name__ == "__main__":
    main()