import numpy as np
import cv2
from scipy.optimize import least_squares
from utils import *

# Performing linear triangulation to get the 3D point set from pixel coordinates
def linear_triangulation(x1,x2,C1,C2,R1,R2,K1,K2):
    X = []
    P1 = projection_matrix(K1,R1,C1)
    P2 = projection_matrix(K2,R2,C2)
    p11, p12, p13 = P1[0,:].reshape((1,4)), P1[1,:].reshape((1,4)), P1[2,:].reshape((1,4))
    p21, p22, p23 = P2[0,:].reshape((1,4)), P2[1,:].reshape((1,4)), P2[2,:].reshape((1,4))
    for i in range(x1.shape[0]):
        A = []
        x1_x, x1_y = x1[i,0], x1[i,1]
        x2_x, x2_y = x2[i,0], x2[i,1]
        A.append((x1_y*p13)-p12)
        A.append(p11-(x1_x*p13))
        A.append((x2_y*p23)-p22)
        A.append(p21-(x2_x*p23))
        A = np.array(A).astype(np.float32).reshape((4,4))
        U, S, V = np.linalg.svd(A)
        Xi = V.T[:,-1]
        X.append(Xi)
    X = np.array(X).astype(np.float32).reshape((-1,4))
    X = X/X[:,3]
    return X # 3D coordinates as shape [N,4], Each coordinate is of the form [x, y, z, 1]

# Reprojection loss between actual feature locations and reprojected (pixel --> 3D --> pixel) point locations
def reprojection_loss(X,x1,x2,P1,P2):
    p11, p12, p13 = P1[0,:].reshape((1,4)), P1[1,:].reshape((1,4)), P1[2,:].reshape((1,4))
    p21, p22, p23 = P2[0,:].reshape((1,4)), P2[1,:].reshape((1,4)), P2[2,:].reshape((1,4))
    # Reprojection loss for the first camera's image
    u1, v1 = x1[0], x1[1]
    u1_proj = np.divide(np.dot(p11,X),np.dot(p13,X))
    v1_proj = np.divide(np.dot(p12,X),np.dot(p13,X))
    error1 = np.square(v1-v1_proj) + np.square(u1-u1_proj)
    # Reprojection loss for the second camera's image
    u2, v2 = x2[0], x2[1]
    u2_proj = np.divide(np.dot(p21,X),np.dot(p23,X))
    v2_proj = np.divide(np.dot(p22,X),np.dot(p23,X))
    error2 = np.square(v2-v2_proj) + np.square(u2-u2_proj)
    # Returning the total error
    return np.squeeze(error1 + error2)

# Non-linear triangulation using least squares optimization to refine the 3D point locations
def non_linear_triangulation(X,x1,x2,C1,C2,R1,R2,K1,K2):
    X_optimized = []
    P1 = projection_matrix(K1,R1,C1)
    P2 = projection_matrix(K2,R2,C2)
    for i in range(X.shape[0]):
        Xi_opt = least_squares(fun=reprojection_loss, x0=X[i,:], method="trf", args=[x1[i,:],x2[i,:],P1,P2])
        Xi_opt = Xi_opt.x
        X_optimized.append(Xi_opt)
    X_optimized = np.array(X_optimized).astype(np.float32).reshape((-1,4))
    X_optimized = X_optimized/X_optimized[:,3]
    return X_optimized # 3D coordinates as shape [N,4], Each coordinate is of the form [x, y, z, 1]