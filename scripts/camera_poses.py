import numpy as np
from utils import *

# Getting the four sets of possible camera center configurations as R,C (R: 3*3 rotation matrix, C:3*1 translation vector)
def get_all_camera_poses(F,K1,K2):
    E = essential_matrix(F,K1,K2)
    U, S, V = np.linalg.svd(E)
    W = np.array([[0,-1, 0],[1, 0, 0],[0, 0, 1]])
    C1 = U[:,2]
    C2 = -U[:,2]
    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W.T,V))
    camera_poses = [[R1, C1], [R2, C1], [R1, C2], [R2, C2]]
    return camera_poses

# Check if points have positive z values, Cheirality Condition
def cheiralityCount(points,T,R_z):
    count = 0
    for point in points:
        if R_z.dot(point.reshape(-1, 1) - T) > 0 and point[2] > 0:
            count += 1
    return count

# Getting the best camera pose (R,C) by checking which of the best of the four poses has most points satisfying the +Z and Chierality Condition
def best_camera_pose(X,camera_poses):
    best_count = 0
    best_index = 0
    for i, R_C in enumerate(camera_poses):
        R, C = R_C
        current_count = cheiralityCount(X,C.reshape((3,1)),R[2,:].reshape((1,3)))
        if current_count > best_count:
            best_count = current_count
            best_index = i
    R_best, C_best = camera_poses[best_index]
    return R_best, C_best