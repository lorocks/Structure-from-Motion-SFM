import numpy as np 
from scipy.optimize import least_squares
from utils import *

# Getting the R, C for a particilar set of 3D (X) <--> 2D (x) point correspondences
def linear_pnp(X,x,K):
    # Making sure X (3D points) are in homogenous coordinates
    if X.shape[1] == 4:
        X = X/(X[:,3].reshape((-1,1)))
    else:
        X = np.hstack((X,np.ones((X.shape[0],1))))
    # Making sure x (2D pixel) locations are in homogenous coordinates
    if x.shape[1] == 3:
        x = x/(x[:,2].reshape((-1,1)))
    else:
        x = np.hstack((x,np.ones((x.shape[0],1))))
    K_inv = np.linalg.inv(K)
    x_img_plane = np.dot(K_inv,x.T).T
    A = None
    for i in range(X.shape[0]):
        X_i = X[i].reshape([1,4]).astype(np.float32)
        zero_vector = np.zeros((1,4))
        u, v, _ = list(np.squeeze(x_img_plane[i]))
        u_cross = np.array([[0,-1,v],[1,0 ,-u],[-v,u,0]])
        X_tilde = np.vstack((np.hstack([X_i,zero_vector,zero_vector]),
                             np.hstack([zero_vector,X_i,zero_vector]),
                             np.hstack([zero_vector,zero_vector,X_i])))
        if i==0:
            A = np.dot(u_cross,X_tilde)
        else:
            A = np.vstack((A,np.dot(u_cross,X_tilde)))
    A = np.array(A)
    U_A, S_A, V_A = np.linalg.svd(A)
    P = V_A[-1].reshape((3,4))
    R, C = P[:,:3], P[:,3]
    U_R, S_R, V_R = np.linalg.svd(R)
    R = np.dot(U_R,V_R)
    if np.linalg.det(R) > 0:
        C = - np.linalg.inv(R).dot(C)
    else:
        R = -R
        C = np.linalg.inv(R).dot(C)
    C = -np.linalg.inv(R).dot(C)
    return R, C

# Reprojection Error while performing PnP Ransac
# X should be of [N,4] and x of [N,2]
def linear_pnp_error(X,x,K,R,C):
    P = projection_matrix(K,R,C)
    p1, p2, p3 = P[0,:].reshape((1,4)), P[1,:].reshape((1,4)), P[2,:].reshape((1,4))
    u, v = x[0], x[1]
    u_proj = np.divide(np.dot(p1,X),np.dot(p3,X))
    v_proj = np.divide(np.dot(p2,X),np.dot(p3,X))
    error = np.squeeze((v-v_proj) + (u-u_proj))
    return float(np.linalg.norm(error))

# Using PnP Ransac to get the best set of R, C using linear PnP
def pnp_ransac(X,x,K,error_thresh=10,n_iter=1000):
    best_num_inliers = 0
    best_R, best_C = None, None
    for i in range(n_iter):
        random_indices = np.random.choice(X.shape[0],size=6)
        X_i, x_i = X[random_indices], x[random_indices]
        R, C = linear_pnp(X_i,x_i,K)
        current_num_inliers = 0
        for j in range(X.shape[0]):
            error = linear_pnp_error(X[j],x[j],K,R,C)
            if error < error_thresh:
                current_num_inliers += 1
        if current_num_inliers > best_num_inliers:
            best_num_inliers = current_num_inliers
            best_R, best_C = R, C
    return best_R, best_C

# Non-Linear PnP Loss Function
# XX = [Q[0], Q[1], Q[2], Q[3], C[0], C[1], C[2]] --> Q is the quaternion and C is the translation vector of the camera center
def non_linear_pnp_loss(XX,X,x,K):
    Q, C = XX[:4], XX[4:].reshape(-1,1)
    R = quaternion_to_rotation(Q)
    P = projection_matrix(K,R,C)
    p1, p2, p3 = P[0,:].reshape((1,4)), P[1,:].reshape((1,4)), P[2,:].reshape((1,4))
    total_error = 0
    for i in range(len(X)):
        X_i, x_i = X[i].reshape((4,1)), x[i].reshape((1,2))
        u, v = x_i[0,0], x_i[0,1]
        u_proj = np.divide(np.dot(p1,X_i),np.dot(p3,X_i))
        v_proj = np.divide(np.dot(p2,X_i),np.dot(p3,X_i))
        total_error += np.squeeze(np.square(v-v_proj) + np.square(u-u_proj))
    return total_error

# Refines the estimate of R, C given the PnP Ransac by using non-linear optimzation
def non_linear_pnp(X,x,K,R_init,C_init):
    Q_init = rotation_to_quaternion(R_init)
    XX_init_quat = np.array([Q_init[0],Q_init[1],Q_init[2],Q_init[3],C_init[0],C_init[1],C_init[2]])
    XX_opt = least_squares(fun=non_linear_pnp_loss,x0=XX_init_quat,method="trf",args=[X,x,K])
    XX_opt = XX_opt.x
    Q_opt, C_opt = XX_opt[:4], XX_opt[4:]
    R_opt = quaternion_to_rotation(Q_opt)
    return R_opt, C_opt
