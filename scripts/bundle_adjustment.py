import numpy as np
from scipy.optimize import least_squares
import scipy.spatial
from scipy.spatial.transform import Rotation 
from scipy.sparse import lil_matrix

def project(points_3d, camera_params, K):
    """
    Project 3D points to 2D using camera parameters

    Inputs:
    points_3d: 3D points to be projected [N,3]
    camera_params: Camera parameters [N,9]

    Outputs:
    points_proj: Projected 2D points [N,2]
    """
    
    def project_point(R, C, pts_3D, K):
        # Project the 3D points to 2D
        # Projection formula
        pts_2d = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        # Pad the 3D points with 1s to make it [N,4]
        x3D_4 = np.hstack((pts_3D, 1))
        x2D = np.dot(pts_2d, x3D_4.T)
        x2D /= x2D[2]
        return x2D
    
    x_proj = []
    for i in range(len(camera_params)):
        R = Rotation.from_rotvec(camera_params[i][:3]).as_matrix()
        C = camera_params[i][3:].reshape(3,1)
        for j in range(len(points_3d)):
            point_3d_j = points_3d[j]
            point_2d_j = project_point(R, C, point_3d_j, K)[:2]
            x_proj.append(point_2d_j)

    # print("Projected 3D Points to 2D Points:", x_proj)
    # print("Shape of the Projected 2D Points: ",np.array(x_proj).shape)
    return np.array(x_proj)

def bundle_adjustment_sparsity(X, camera_num):
    """
    Create the sparsity matrix for the bundle adjustment problem

    Inputs:
    X: 3D points [N,3]
    camera_num: Number of cameras

    Outputs:
    sparsity: Sparsity matrix 
    """

    # Increment the camera_num by 1 to adjust for 0-based indexing
    number_of_cameras = camera_num+1
    num_points = X.shape[0]

    m = num_points*2
    n = number_of_cameras*6 + num_points*3

    sparsity = lil_matrix((m, n), dtype=int)

    # Fill the sparsity matrix
    for i in range(num_points):
        for s in range(3):
            sparsity[2*i, number_of_cameras*6 + 3*i + s] = 1
            sparsity[2*i+1, number_of_cameras*6 + 3*i + s] = 1

    return sparsity


def fun(x0, camera_num, num_3d_points, points_2d, K):
    """
    Function to be optimized

    Inputs:
    params: Parameters to be optimized [N,9+3], contains camera parameters and 3D points
    camera_num: Number of cameras
    num_3d_points: Number of 3D points  (N)
    points_2d: 2D points [N,2], i.e., the feature matrix

    Outputs:
    residuals: Residuals [N*2]
    """

    # Increment the camera_num by 1 to adjust for 0-based indexing
    number_of_cameras = camera_num+1
    # number_of_cameras = camera_num
    # print("Number of Cameras: ",number_of_cameras)

    # Get the camera parameters
    camera_params = x0[:number_of_cameras*6].reshape((number_of_cameras, 6))
    # print("Camera Parameters: ",camera_params)
    # print("Shape of Camera Parameters: ",camera_params.shape)

    # Get the 3D points
    points_3d = x0[number_of_cameras*6:].reshape((num_3d_points, 4))

    # Shave off the last column of 1s
    points_3d = points_3d[:,:3]

    # Project the 3D points to 2D
    points_proj = project(points_3d, camera_params, K)
    # print("Shape of camera_params: ", camera_params.shape)
    # points_proj = project(points_3d, camera_params)

    # Consider only the 2D points from the 0th to the (number_of_cameras-1)th camera
    points_2d = points_2d[:num_3d_points*number_of_cameras]
    
    # Calculate the residuals
    residuals = (points_proj - points_2d).ravel()

    return residuals


# Perform Bundle Adjustment to Improve the Estimate of the Reconstructed 3D point locations
def bundle_adjustment(img_idx, points_3d, feature_matrix, all_RC, K):
    """
    Perform Bundle Adjustment to improve the estimate of the reconstructed 3D point locations
    Assume that every 3D point is visible in every image

    Inputs:
    img_idx: The index of the image for which the 3D points are to be optimized
    X: 3D points common to all images (to be optimized)
    feature_matrix: [N,M,2] shape matrix containing the pixel coordinates of the features in all images
    (N: Number of images, M: Number of features in each image, 2: x and y coordinates of the features)
    all_RC: List of camera poses for all images registered so far
    K: Camera Intrinsic Matrix

    Outputs:
    X: Optimized 3D points
    """

    # Get a list of all the camera parameters for all images
    RC_list = []

    for i in range(len(all_RC)):
        R, C = all_RC[i]
        # Convert Rotation Matrix to Rotation Vector
        R_vec = Rotation.from_matrix(R).as_rotvec()
        # Create a list of camera parameters
        RC_i = [R_vec[0], R_vec[1], R_vec[2], C[0], C[1], C[2]]
        RC_list.append(RC_i)


    RC_list = np.array(RC_list).reshape(-1, 6)

    # Set the initial parameters for the optimization
    x0 = np.hstack((RC_list.flatten(), points_3d.flatten()))
    # print("Initial Parameters: ",x0)
    # print("Shape of the Initial Parameters: ",x0.shape)
    
    # Get the number of 3D points
    num_points = points_3d.shape[0]
    # print("Number of 3D Points: ",num_points)

    # It should be noted that we have assumed that every 3D point is visible in every image

    # sparsity = bundle_adjustment_sparsity(points_3d, img_idx)
    # sparsity = np.ones((2*num_points, 6*(img_idx+1) + 3*num_points))

    # Setup the optimization problem
    res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
                        args=(img_idx, num_points, feature_matrix.reshape(-1, 2), K)) # twas ftol = 1e-10

    # res = least_squares(fun, x0, jac_sparsity=sparsity, verbose=2, args=(img_idx, num_points, feature_matrix.reshape(-1, 2), K))

    params = res.x
    print(len(params))
    num_cameras = img_idx+1
    optimized_RC = params[:num_cameras*6].reshape((num_cameras, 6))
    optimized_3D = params[num_cameras*6:].reshape((num_points, 3))

    optimized_C, optimized_R = [], []
    for i in range(num_cameras):
        R = Rotation.from_rotvec(optimized_RC[i][:3]).as_matrix()
        C = optimized_RC[i][3:].reshape(3,1)
        optimized_C.append(C)
        optimized_R.append(R)

    return optimized_R, optimized_C, optimized_3D
