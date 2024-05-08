import os
import numpy as np
import cv2

# Finding the keypoint locations of the matched features of two images using their SIFT features
def find_matches(keypoints1,descriptors1,keypoints2,descriptors2,matching_thresh=0.95):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < matching_thresh*n.distance:
            good_matches.append(m)
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return matched_keypoints1, matched_keypoints2

# Finding all keypoint locations of the feature locations common to all of the images
def find_common_features(images_dir,reference_img_name):
    image_files = os.listdir(images_dir)
    sift = cv2.SIFT_create()
    reference_img = cv2.imread(os.path.join(images_dir,reference_img_name))
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_gray,None)
    matched_features_dict = {}
    for img_name in image_files:
        if img_name == reference_img_name:
            continue
        img = cv2.imread(os.path.join(images_dir,img_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_keypoints, img_descriptors = sift.detectAndCompute(img_gray,None)
        matched_keypoints_ref, matched_keypoints_img = find_matches(reference_keypoints,reference_descriptors,img_keypoints,img_descriptors)
        for kp_ref, kp_img in zip(matched_keypoints_ref, matched_keypoints_img):
            feature_name = tuple(np.round(kp_ref).astype(int))
            if feature_name not in matched_features_dict:
                matched_features_dict[feature_name] = {}
            matched_features_dict[feature_name][img_name] = kp_img
    kp_delete = []
    for kp_ref in matched_features_dict:
        if len(matched_features_dict[kp_ref]) != len(image_files)-1:
            kp_delete.append(kp_ref)
    for kp_ref in kp_delete:
        del matched_features_dict[kp_ref]
    return matched_features_dict

# Converting the dictionary of common feature keypoint locations to a numpy array
def construct_feature_matrix(images_dir,reference_img_name):
    image_files = os.listdir(images_dir)
    matched_feature_dict = find_common_features(images_dir,reference_img_name)
    N = len(matched_feature_dict.items())
    feature_matrix = [[] for i in range(len(image_files))]
    for kp_ref in matched_feature_dict.keys():
        feature_matrix[image_files.index(reference_img_name)].append([kp_ref])
        for img_name in image_files:
            if img_name == reference_img_name:
                continue
            else:
                kp_img = matched_feature_dict[kp_ref][img_name]
                feature_matrix[image_files.index(img_name)].append([kp_img])
    feature_matrix = np.array(feature_matrix).reshape(len(image_files),-1,2)
    return feature_matrix # feature_matrix has a shape [N,M,2] --> N: total nnumber of images, M: total number of common features