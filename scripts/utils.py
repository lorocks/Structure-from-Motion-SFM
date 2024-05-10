import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Loading all the image data
def load_imgs(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        images.append(img)
    return images

'''
# Detecting feature locations and descriptors
def detect_BRISK_features(rgb_img):
    BRISK = cv2.BRISK_create()
    gray_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
    kp, features = BRISK.detectAndCompute(gray_img, None)
    return kp, features

# Matching features from one image to another to get a set of matched points and matrix
def match_BRISK_features(kp1,features1,kp2,features2):
    BF = cv2.BFMatcher()
    matches = BF.knnMatch(features1,features2,k=2)
    img1_pts, img2_pts, linking_matrix = [], [], []
    for m,n in matches:
        (x1,y1) = kp1[m.queryIdx].pt
        (x2,y2) = kp2[m.trainIdx].pt
        img1_pts.append([x1,y1])
        img2_pts.append([x2,y2])
        linking_matrix.append([x1,y1,x2,y2])
    img1_pts = np.array(img1_pts)
    img2_pts = np.array(img2_pts)
    linking_matrix = np.array(linking_matrix)
    return img1_pts, img2_pts, linking_matrix
'''

# Creating a linking matrix using the keypoints of matched features of two images
def create_linking_matrix(img1_pts,img2_pts):
    linking_matrix = []
    for i in range(img1_pts.shape[0]):
        linking_matrix.append([img1_pts[i,0],img1_pts[i,1],img2_pts[i,0],img2_pts[i,1]])
    linking_matrix = np.array(linking_matrix)
    return linking_matrix

# Computing the essential matrix from the Fundamental Matrix and Intrinsic Camera Parameters
def essential_matrix(F,K1,K2):
    E = np.dot(K2.T,np.dot(F,K1))
    U, S, V = np.linalg.svd(E)
    S = np.diag([1,1,0])
    E = np.dot(U,np.dot(S,V))
    return E

# Computing the Projection Matrix (x = PX  --> P = K[R,T])
def projection_matrix(K,R,C):
    P = np.dot(K,np.dot(R,np.hstack([np.eye(3),-C.reshape((3,1))])))
    return P

# Converting a rotation matrix to a quaternion
def rotation_to_quaternion(R):
    Q = Rotation.from_matrix(R).as_quat()
    return Q

# Converting a quaternion to a rotation matrix
def quaternion_to_rotation(Q):
    R = Rotation.from_quat(Q).as_matrix()
    return R

# Reprojection loss between actual feature locations and reprojected (pixel --> 3D --> pixel) point locations
# X should be of shape [N,4] and x of [N,2]
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

# X should be of [N,4] and x of [N,2]
def linear_pnp_error(X,x,K,R,C):
    P = projection_matrix(K,R,C)
    p1, p2, p3 = P[0,:].reshape((1,4)), P[1,:].reshape((1,4)), P[2,:].reshape((1,4))
    u, v = x[0], x[1]
    u_proj = np.divide(np.dot(p1,X),np.dot(p3,X))
    v_proj = np.divide(np.dot(p2,X),np.dot(p3,X))
    error = np.squeeze((v-v_proj) + (u-u_proj))
    return float(np.linalg.norm(error))

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