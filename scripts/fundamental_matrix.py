import numpy as np

# Functions for used to normalize points in the fundamental matrix calculation
def normalize_fmatrix_pts(points):
    point_mean = np.mean(points, axis=0)
    x_mean = point_mean[0]
    y_mean = point_mean[1]
    x_hat = points[:,0] - x_mean
    y_hat = points[:,1] - y_mean
    dist = np.mean(np.sqrt(x_hat**2 + y_hat**2))
    S = np.diag([(2/dist),(2/dist),1])
    T = np.dot(S, np.array([[1,0,-x_mean],[0,1,-y_mean],[0,0,1]]))
    X = np.column_stack((points, np.ones(8)))
    return  (T.dot(X.T)).T, T

# Calculating the fundamental matrix from point correspondences
def findFMatrix(points):
    matrix = np.array([])
    # points_norm, T1, T2 = normalize(points)
    x1 = points[:,0:2]
    x2 = points[:,2:4]
    x1, T1 = normalize_fmatrix_pts(x1)
    x2, T2 = normalize_fmatrix_pts(x2)
    points_norm = points.copy()
    points_norm[:, 0:2] = x1[:, 0:2]
    points_norm[:, 2:4] = x2[:, 0:2]
    # Iterate over all points for homography calculation
    for x2, y2, x1, y1 in points_norm:
        matrix = np.append(matrix, [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    matrix = matrix.reshape(8, -1)
    # Calculate SVD
    _, s, v = np.linalg.svd(matrix)
    eigen = v[-1:]
    F = np.reshape(eigen, (3,3))
    uF, sF, vF = np.linalg.svd(F)
    sF = np.diag(sF)
    sF[2, 2] = 0
    F = np.dot(uF, np.dot(sF, vF))
    F = np.dot(T2.T, np.dot(F, T1))
    return F # Returns 3x3 np array