import numpy as np
import math
from fundamental_matrix import *

# Calculate error between actual points and homographical transformed points
def find_fmatrix_ransac_error(points,F,threshold):
  below_thresh = []
  count = 0
  for x1, y1, x2, y2 in points:
    point1 = np.array([[x1], [y1], [1]]).T
    point2 = np.array([[x2], [y2], [1]])
    # Calculate homographical transform
    calc = np.abs(np.dot(point1, np.dot(F, point2)))
    if calc < threshold: # APPEND
      below_thresh.append([x1, y1, x2, y2])
      count += 1
  return count, np.array(below_thresh) # Returns count of best inliers in RANSAC, and, the best_matches array

# RANSAC algorithm to get the best Fundamental Matrix relating two images
def FMatrix_RANSAC(points,sample_size,threshold):
    # Set maximum iteration number
    max_iteration = 1000
    # Set minimum iteration number
    min_iteration = 300
    # Iteration counter
    iteration = 0
    # Store count of maximum number of inliers
    inlier_count_max = 0
    # Store best fit homography model
    best_fit = None
    best_inliers = []
    # Set probabilities
    prob_outlier = 0.5
    prob_final = 0.95
    n = len(points)
    inlier_count = 1
    # Variables used to ensure RANSAC doesn't quit randomly
    long_iter = 0
    long_iter_count = 0
    old_inliers = 0
    # Loop while iteration number is below maximum iterations
    while iteration < max_iteration:
        # Count longest number of iterations where maximum number of inliers doesn't change
        # Used to ensure proper code execution
        if old_inliers == inlier_count_max:
            long_iter_count += 1
        else:
            if long_iter_count > long_iter:
                long_iter = long_iter_count
                long_iter_count = 0
        old_inliers = inlier_count_max
        # Get 8 random points
        np.random.shuffle(points)
        samples = points[:sample_size]
        # Find homography
        F_check = findFMatrix(samples)
        # Get total number of inliers
        inlier_count, inlier_points = find_fmatrix_ransac_error(points, F_check, threshold)
        # Check if current inliers are greater than previous one and update variables
        if inlier_count > inlier_count_max:
            inlier_count_max = inlier_count
            best_fit = F_check
            best_inliers = inlier_points
            prob_outlier = 1-(inlier_count/n)
        if prob_outlier > 0:
            # Recompute maximum iterations
            try:
                max_iteration = math.log(1-prob_final)/math.log(1-(1-prob_outlier)**sample_size)
            except:
                max_iteration = math.log(1-prob_final)/0.00001
            # Logic to ensure RANSAC runs properly and only stops execution by incrementally reaching maximum iterations
            if iteration > max_iteration:
                if long_iter_count > long_iter:
                    long_iter = long_iter_count
                    long_iter_count = 0
                max_iteration += long_iter
                long_iter -= 1
            # Ensure minimum iteration take place
            if iteration < min_iteration and max_iteration < min_iteration:
                max_iteration = min_iteration
        max_iteration = int(max_iteration)
        print("Max Iterations:", max_iteration, "Iteration Num:", iteration, "Inlier Count:", inlier_count_max, "Long Iter Value:", long_iter, "Long Iter Count:", long_iter_count)
        iteration+=1
        if iteration > 500:
            break
    #print(inlier_count_max, len(best_inliers))
    return best_fit, best_inliers # Returns 3x3 np array, and Nx4 np array