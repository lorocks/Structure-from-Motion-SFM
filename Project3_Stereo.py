import numpy as np
import cv2
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


### Functions for Section 3.1
def normalize(points):
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

def findFMatrix(points):
  matrix = np.array([])
  # points_norm, T1, T2 = normalize(points)

  x1 = points[:,0:2]
  x2 = points[:,2:4]
  x1, T1 = normalize(x1)
  x2, T2 = normalize(x2)
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

# Calculate error between actual points and homographical transformed points
def findError(points, F, threshold):
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


# RANSAC algorithm
def RANSAC(points, sample_size, threshold):
  # Set maximum iteration number
  max_iteration = 1000
  # Set minimum iteration number
  min_iteration = 1000
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
    inlier_count, inlier_points = findError(points, F_check, threshold)

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
    # print("Max Iterations:", max_iteration, "Iteration Num:", iteration, "Inlier Count:", inlier_count_max, "Long Iter Value:", long_iter, "Long Iter Count:", long_iter_count)
    iteration+=1
    if iteration > 10000:
      break

  print(inlier_count_max, len(best_inliers))
  return best_fit, best_inliers # Returns 3x3 np array, and Nx4 np array

### Section 3.1 Functions End


### Function to Display Epipoles
def getMinMaxXY(x_min, x_max, y_min, y_max, line_info):
  if x_min == 0:
    y_min = y_max = -line_info[2]/line_info[1]
  else:
    x_min = -((line_info[1] * y_min) + line_info[2])/line_info[0]
    x_max = -((line_info[1] * y_max) + line_info[2])/line_info[0]

  return x_min, x_max, y_min, y_max


def drawEpipoles(points, F, image):
  draw_image = image.copy()
  for ppoint in points:
    point = np.array([[ppoint[0]], [ppoint[1]], [1]])

    line = np.dot(F, point)

    x_min, x_max, y_min, y_max = getMinMaxXY(None, None, 0, image.shape[0], line)

    cv2.circle(draw_image, (int(point[0][0]),int(point[1][0])), 10, (0,0,255), -1)

    cv2.line(draw_image, (int(x_min[0]), int(y_min)), (int(x_max[0]), int(y_max)), (0, 255, 0), 2)

  return draw_image

def drawRectEpipoles(points, F, image):
  draw_image = image.copy()
  for ppoint in points:
    point = np.array([[ppoint[0]],[ppoint[1]], [1]])

    line = np.dot(F, point)

    x_min, x_max, y_min, y_max = getMinMaxXY(0, image.shape[1] - 1, None, None, line)

    cv2.circle(draw_image, (int(point[0][0]),int(point[1][0])), 10, (0,0,255), -1)

    cv2.line(draw_image, (int(x_min), int(y_min[0])), (int(x_max), int(y_max[0])), (0, 255, 0), 2)

  return draw_image

### Function to Display Epipoles End


### Function for Section 3.5
# Check if points have positive z values, Cheirality Condition
def cheiralityCount(points ,T, R_z):
    count = 0
    for point in points:
        if R_z.dot(point.reshape(-1, 1) - T) > 0 and point[2] > 0:
            count += 1
    return count

### Function for Section 3.5 End

# Function to calculate SSD and then fill a disparity array
def disparitySSD(image_left, image_right):
     gray1 = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
     gray2 = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
     window_size = 30
     block = 7

     disparity = np.zeros(gray1.shape)

     # Calculate ssd between images
     # Compares a block in image 1 within a window range in image 2
     for i in tqdm(range(block, gray1.shape[0] - block - 1)):
          for j in range(block + window_size, gray1.shape[1] - block - 1):
               ssd = np.empty([window_size,1])
               l = gray1[(i-block):(i+block), (j-block):(j+block)]
               for f in range(0,window_size):
                    r = gray2[(i-block):(i+block), (j-f-block):(j-f+block)]
                    ssd[f] = np.sum((l[:,:] - r[:,:])**2)
               disparity[i,j] = np.argmin(ssd)

     return disparity


# Function to calculate SAD and then fill a disparity array
def disparitySAD(image_left,image_right):
    block = 3
    local_window = 5

    left = np.asarray(image_left).astype(int)
    right = np.asarray(image_right).astype(int)

    height, width, _ = left.shape

    disparity = np.zeros((height, width))
    #going over each pixel
    for y in tqdm(range(block, height-block)):
        for x in range(block, width-block):
            left_local = left[y:y + block, x:x + block]

            x_min = max(0, x - local_window)
            x_max = min(right.shape[1], x + local_window)
            min_sad = None
            index_min = None
            first = True

            for x in range(x_min, x_max):
                right_local = right[y: y+block,x: x+block]
                if left_local.shape == right_local.shape:
                  sad = np.sum(abs(left_local - right_local))
                else:
                  sad = -1
                if first:
                    min_sad = sad
                    index_min = (y, x)
                    first = False
                else:
                    if sad < min_sad:
                        min_sad = sad
                        index_min = (y, x)

            disparity[y, x] = abs(index_min[1] - x)

    return disparity


### Functions to Rectify Epipolar Lines
# Calculate epipole from Fundamental matrix
def getEpipole(F):
   u, s, v = np.linalg.svd(F)
   eigen = v[-1, :]
   return eigen / eigen[2]

# Calculate Homography of Rectification
def computeRectificationHomography(e, height, width):
    T = np.array([[1, 0, -width/2], [0, 1, -height/2], [0, 0, 1]])
    e_p = T @ e
    e_p = e_p / e_p[2]
    ex = e_p[0]
    ey = e_p[1]

    if ex >= 0:
        a = 1
    else:
        a = -1

    R1 = a * ex / np.sqrt(ex ** 2 + ey ** 2)
    R2 = a * ey / np.sqrt(ex ** 2 + ey ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e_p = R @ e_p
    x = e_p[0]

    G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])

    H = np.linalg.inv(T) @ G @ R @ T

    return H

### Functions End

def main():
  # Read images
  image1 = cv2.imread(f"im0.png")
  image2 = cv2.imread(f"im1.png")

  # Calibration Text Files
  # Get data from calib.txt
  f = open(f"calib.txt", 'r')
  for line in f:
    line_list = line.split('=')
    if line_list[0] == "cam0":
      values = line_list[1][1:-2].split(' ')
      K0 = []
      row = []

      for value in values:
        if value[-1] == ';':
          row.append(float(value[:-1]))
          K0.append(row)
          row = []
        else:
          row.append(float(value))
      K0.append(row)
      K0 = np.array(K0)
      print("\nIntrinsic Matrix of Camera 1:")
      print(K0)

    elif line_list[0] == "cam1":
      values = line_list[1][1:-2].split(' ')
      K1 = []
      row = []

      for value in values:
        if value[-1] == ';':
          row.append(float(value[:-1]))
          K1.append(row)
          row = []
        else:
          row.append(float(value))
      K1.append(row)
      K1 = np.array(K1)
      print("\nIntrinsic Matrix of Camera 2:")
      print(K1)

    elif line_list[0] == "baseline":
      baseline = float(line_list[1].split(' ')[0])
      print("\nBaseline:")
      print(baseline)


  ### Section 3.1 Start
  # Define BRISK feature extractor
  BRISK = cv2.BRISK_create()

  # Define Brute Force matcher
  BF = cv2.BFMatcher()

  # Convert to grayscale
  gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # Find keypoints and descriptors
  keypoints1, descriptors1 = BRISK.detectAndCompute(gray_image1, None)
  keypoints2, descriptors2 = BRISK.detectAndCompute(gray_image2, None)

  # Display keypoints
  keypoint_img = cv2.drawKeypoints(image1, keypoints1, None, color = (0,255,0), flags = 0)
  keypoint_img = cv2.drawKeypoints(image2, keypoints2, None, color = (0,255,0), flags = 0)

  # Perform Brute Force matching
  matches_display = BF.knnMatch(descriptors1, descriptors2, k=2)

  # Creating array in the form [point1, point2] for homography calculation
  linking_matrix = []
  view_matches = []
  for m,n in matches_display:
      if m.distance < 0.7*n.distance:
          (x1, y1) = keypoints1[m.queryIdx].pt  # List 1
          (x2, y2) = keypoints2[m.trainIdx].pt  # List 2
          view_matches.append([m])
          linking_matrix.append([x1, y1, x2, y2])
  linking_matrix = np.array(linking_matrix) # A Nx4 np array, 
  # Column 1 contains x position of feature 1, 
  # Column 2 contains y position of feature 1, 
  # Column 3 contains x position of feature 2, 
  # Column 4 contains y position of feature 2

  # Display matches
  match_img = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, view_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  ### Section 3.1 End


  ### Section 3.2 Start
  # Perform RANSAC
  # F is a 3x3 np array
  # best_matches  is a subset of linking_matrx, 4 columns and rows are based on how many best inliers are found in RANSAC
  # best_matches[:, 0:2] => gives best inlier features for image 1, N/2 x 4
  # best_matches[:, 2:4] => gives best inlier features for image 2, N/2 x 4
  F, best_matches = RANSAC(linking_matrix.copy(), 8, 0.002)
  print("\nFundamental Matrix:")
  print(F)

  ### Section 3.2 End


  ### Section 3.3 Start
  # # E matrix
  # E is a 3x3 np array
  E = np.dot(K1.T, np.dot(F, K0))

  # Reduce rank
  u, s, v = np.linalg.svd(E)
  s = [1,1,0]
  E = np.dot(u, np.dot(np.diag(s), v))

  print("\nEssential Matrix:")
  print(E) # 3x3 np array

  ### Section 3.3 End


  ### Section 3.4 Start
  # Calculate poses
  u, s, v = np.linalg.svd(E)
  W = np.array([[0,-1, 0],
                [1, 0, 0],
                [0, 0, 1]])
  T1 = u[:, 2]
  T2 = -T1
  R1 = np.dot(u, np.dot(W, v))
  R2 = np.dot(u, np.dot(W.T, v))

  poses = [[R1, T1], [R2, T1], [R1, T2], [R2, T2]]

  # T is actually C which is a 1x3 np array
  # R is 3x3 np array

  ### Section 3.4 End


  ### Section 3.5 Start
  # Linear Trianglulation
  points_3D = []
  R_best = None
  T_best = None
  max_num = 0

  for pose in poses:
    if np.linalg.det(pose[0]) < 0:
      pose[0] = -pose[0]
      pose[1] = -pose[1]

    R = pose[0]
    T = pose[1]

    P1 = np.dot(K0, np.hstack((np.identity(3), np.zeros((3, 1)))))
    P2 = np.dot(K1, np.hstack((R, -T.reshape(3, 1))))

    for point in best_matches:
      triangle_points = cv2.triangulatePoints(P1, P2, np.float32(point[0:2]), np.float32(point[2:4]))
      points_3D.append(np.array(triangle_points)[0:3, 0])

  # Cheirality Condiiton
  for pose in poses:
    current_num = cheiralityCount(points_3D, pose[1].reshape(3, 1), pose[0][2].reshape(1, 3))

    if current_num > max_num:
      R_best = pose[0]
      T_best = pose[1]
      max_num = current_num

  print("\nRotation Matrix:")
  print(R_best)
  print("\nTranslation:")
  print(T_best)

  ### Section 3.5 End




  # Draw Epipolar lines before rectification
  epipole_image1 = drawEpipoles(best_matches[:, 0:2], F, image1)
  epipole_image2 = drawEpipoles(best_matches[:, 2:4], F, image2)

  combine_image = np.concatenate((epipole_image1, epipole_image2), axis = 1)

  ##### Manual Calculation of Homographies
  e = getEpipole(F.T) # Returns a singular value
  H2 = computeRectificationHomography(e, image2.shape[0], image2.shape[1]) # Returns 3x3 np array

  # Computing Homographies ##### Uses OpenCV
  ##### Finding H2 works, but doing RANSAC for H1 does give very good results yet
  _, H1, H2_ = cv2.stereoRectifyUncalibrated(np.float32(best_matches[:, 0:2]), np.float32(best_matches[:, 2:4]), F, imgSize=(image1.shape[1], image1.shape[0]))
  print("Estimated H1 and H2 as \n \n Homography Matrix 1: \n", H1,'\n \n Homography Matrix 2:\n ', H2)


  # Get rectified images
  rect_image1 = cv2.warpPerspective(image1, H1, (image1.shape[1], image1.shape[0]))
  rect_image2 = cv2.warpPerspective(image2, H2, (image2.shape[1], image2.shape[0]))

  small_rect_image11 = cv2.resize(rect_image1, (int(rect_image1.shape[1] / 4), int(rect_image1.shape[0] / 4)))
  small_rect_image21 = cv2.resize(rect_image2, (int(rect_image2.shape[1] / 4), int(rect_image2.shape[0] / 4)))


  # Get rectified epipolar points
  pts_set1_rectified = cv2.perspectiveTransform(best_matches[:, 0:2].reshape(-1, 1, 2), H1).reshape(-1,2)
  pts_set2_rectified = cv2.perspectiveTransform(best_matches[:, 2:4].reshape(-1, 1, 2), H2).reshape(-1,2)

  F_rect = np.dot(np.linalg.inv(H2.T), np.dot(F, np.linalg.inv(H1)))

  rect_epipole_image1 = drawRectEpipoles(pts_set1_rectified, F_rect, rect_image1)
  rect_epipole_image2 = drawRectEpipoles(pts_set2_rectified, F_rect, rect_image2)

  # Display rectified epipolar lines
  combine_rect_image = np.concatenate((rect_epipole_image1, rect_epipole_image2), axis = 1)


  # Perform SSD Disparity
  print("Starting Disparity for SSD")
  disp_ssd = disparitySSD(small_rect_image11, small_rect_image21)


  show_disp_ssd = ((disp_ssd/disp_ssd.max())*255).astype(np.uint8)

  # Display Disparity Heat Plot
  plt.imshow(show_disp_ssd, cmap='hot', interpolation='bilinear')
  plt.title('Disparity Plot Heat')
#   plt.savefig(f"{actual_folder}/disparity_heat_ssd.png")
  plt.show()

  # Display Disparity Gray Plot
  plt.imshow(show_disp_ssd, cmap='gray', interpolation='bilinear')
  plt.title('Disparity Plot Gray')
#   plt.savefig(f"{actual_folder}/disparity_gray_ssd.png")
  plt.show()

  disp_ssd += 1

  depth_ssd = (baseline * K1[0, 0] )/disp_ssd

  show_depth_ssd = ((depth_ssd/depth_ssd.max())*255).astype(np.uint8)

  # Display Depth Image as Heat Map
  plt.imshow(show_depth_ssd, cmap='hot', interpolation='bilinear')
  plt.title('Depth Plot Heat')
#   plt.savefig(f"{actual_folder}/depth_heat_ssd.png")
  plt.show()

  # Display Depth Image as Gray Map
  plt.imshow(show_depth_ssd, cmap='gray', interpolation='bilinear')
  plt.title('Depth Plot Gray')
#   plt.savefig(f"{actual_folder}/depth_heat_ssd.png")
  plt.show()


  # Perform SAD Disparity
  print("Starting Disparity for SAD")
  disp_sad = disparitySAD(small_rect_image11, small_rect_image21)


  show_disp_sad = ((disp_sad/disp_sad.max())*255).astype(np.uint8)

  # Display Disparity Heat Plot
  plt.imshow(show_disp_sad, cmap='hot', interpolation='bilinear')
  plt.title('Disparity Plot Heat')
#   plt.savefig(f"{actual_folder}/disparity_heat_sad.png")
  plt.show()

  # Display Disparity Gray Plot
  plt.imshow(show_disp_sad, cmap='gray', interpolation='bilinear')
  plt.title('Disparity Plot Gray')
#   plt.savefig(f"{actual_folder}/disparity_gray_sad.png")
  plt.show()


  # Plot disparity images for SAD
  disp_sad += 1

  depth_sad = (baseline * K1[0, 0] )/disp_sad

  show_depth_sad = ((depth_sad/depth_sad.max())*255).astype(np.uint8)

  # Display Depth Image as Heat Map
  plt.imshow(show_depth_sad, cmap='hot', interpolation='bilinear')
  plt.title('Depth Plot Heat')
#   plt.savefig(f"{actual_folder}/depth_heat_sad.png")
  plt.show()

  # Display Depth Image as Gray Map
  plt.imshow(show_depth_sad, cmap='gray', interpolation='bilinear')
  plt.title('Depth Plot Gray')
#   plt.savefig(f"{actual_folder}/depth_gray_sad.png")
  plt.show()

try:
  disp = main()
except Exception as err:
  print("Invalid input")
  print(err)
