import os
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

# Loading all the image data
def load_imgs(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        images.append(img)
    return images

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

def all_RC_flatten(all_RC):
    flat_list = []
    for i in range(all_RC.shape[0]):
        pass

# A Function to Plot the 3D Point-Cloud
def plot_3D_points(X):
    if X.shape[1] == 4:
        X = X/(X[:,3].reshape((-1,1)))
        X = X[:,:-1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    colors = np.array([[1,0,0] for _ in range(X.shape[0])])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

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