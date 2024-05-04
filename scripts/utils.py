import numpy as np
import cv2

# Computing the essential matrix from the Fundamental Matrix and Intrinsic Camera Parameters
def essential_matrix(F,K1,K2):
    E = np.dot(K2.T,np.dot(F,K1))
    U, S, V = np.linalg.svd(E)
    S = np.diag([1,1,0])
    E = np.dot(U,np.dot(S,V))
    return E

# Computing the Projection Matrix (x = PX  --> P = K[R,T])
def projection_matrix(K,R,C):
    P = np.dot(K,np.dot(R,np.hstack(np.eye(3),-C.reshape((3,1)))))
    return P