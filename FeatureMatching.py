import cv2 as cv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np
import pandas as pd 
import pydicom

# create a dictionary of data 
data = { 'Algo':[], 'Features in Image 1': [],
        'Features in Image 2': [],
        'No. of Matched Features': [],
        'No. of Matched Features after RANSAC': [],
        'Feature Description Computation Time (2 Images) -': [],
        'Feature Matching Computation Time': [] }

''' Parameters
1. #Matches
2. Comp Time 
   a. Feature Extraction & Descriptor Computation
   b. Feature Matching
3. Best Matches'''

df = pd.DataFrame(data)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# img1_path = os.path.join(dir_path, 'VHF_Brain1.png')
# img2_path = os.path.join(dir_path, 'VHF_Brain2.png')


# img1 = cv.imread(img1_path)
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# img2 = cv.imread(img2_path)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Load the DICOM image
ds = pydicom.dcmread('/Users/mukund/Documents/ECE_613_PROJECT/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-001.dcm')
ds1 = pydicom.dcmread('/Users/mukund/Documents/ECE_613_PROJECT/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-130.dcm')

# ds = pydicom.dcmread('/Users/mukund/Documents/ECE_613_PROJECT/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm')
# ds1 = pydicom.dcmread('/Users/mukund/Documents/ECE_613_PROJECT/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-030.dcm')




# Extract the pixel array
pixel_array = ds.pixel_array
pixel_array1 = ds1.pixel_array

# Convert the pixel array to an OpenCV image
cv_image = np.array(pixel_array, dtype=np.uint16)
img1 = cv2.convertScaleAbs(cv_image, alpha=(255.0/65535.0))
cv.imshow('x',img1)
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

cv_image1 = np.array(pixel_array1, dtype=np.uint16)
img2 = cv2.convertScaleAbs(cv_image1, alpha=(255.0/65535.0))
cv.imshow('as',img2)
cv.waitKey(0)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


detector = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
comp_start = time.time()
kp1, des1 = detector.detectAndCompute(img1,None)
kp2, des2 = detector.detectAndCompute(img2,None)
comp_end = time.time()
comp_time = comp_end - comp_start

# BFMatcher with default params
#ToDo - Can also try other matching algorithms - FLANN

bf = cv.BFMatcher()
match_start = time.time()
matches = bf.knnMatch(des1,des2,k=2)
match_end = time.time()
match_time = match_end - match_start
total_matches = len(matches)

# good = matches

# Lowe's Ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Apply RANSAC to find the best set of matches
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

best_src_pts = None
best_dst_pts = None
best_inliers = 0

# Ransac
model, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform, min_samples=4,
        residual_threshold=8, max_trials=10000
    )

n_inliers = np.sum(inliers)
inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
plt.imshow(image3), plt.title('SIFT'), plt.show()

# cv2.imshow('Matches', image3)
# cv2.waitKey(0)

src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

data['Algo'].append('SIFT')
data['Features in Image 1'].append(len(kp1))
data['Features in Image 2'].append(len(kp2))
data['No. of Matched Features'].append(len(good))
data['No. of Matched Features after RANSAC'].append(n_inliers)
data['Feature Description Computation Time (2 Images) -'].append(comp_time)
data['Feature Matching Computation Time'].append(match_time)
# print('Features in Image 1 - ',len(kp1))
# print('Features in Image 2 - ',len(kp2))
# print('No. of Matched Features - ', len(good))
# print('No. of Matched Features after RANSAC - ', n_inliers)
# print('Feature Description Computation Time (2 Images) - ',comp_time)
# print('Feature Matching Computation Time - ',match_time)

surf_detector = cv.xfeatures2d.SURF_create()

# find the keypoints and descriptors with SIFT
comp_start = time.time()
kp1, des1 = surf_detector.detectAndCompute(img1,None)
kp2, des2 = surf_detector.detectAndCompute(img2,None)
comp_end = time.time()
comp_time = comp_end - comp_start

# BFMatcher with default params
#ToDo - Can also try other matching algorithms - FLANN

bf_surf = cv.BFMatcher()
match_start = time.time()
matches = bf_surf.knnMatch(des1,des2,k=2)
match_end = time.time()
match_time = match_end - match_start
total_matches = len(matches)

# Lowe's Ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# Apply RANSAC to find the best set of matches
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

best_src_pts = None
best_dst_pts = None
best_inliers = 0

# Ransac
model, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform, min_samples=4,
        residual_threshold=8, max_trials=10000
    )

n_inliers = np.sum(inliers)
inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
plt.title('SURF'), plt.imshow(image3), plt.show()
# cv2.imshow('Matches', image3)
# cv2.waitKey(0)

src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

data['Algo'].append('SURF')
data['Features in Image 1'].append(len(kp1))
data['Features in Image 2'].append(len(kp2))
data['No. of Matched Features'].append(len(good))
data['No. of Matched Features after RANSAC'].append(n_inliers)
data['Feature Description Computation Time (2 Images) -'].append(comp_time)
data['Feature Matching Computation Time'].append(match_time)

# Create ORB feature detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors for the images
start = time.time()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
end = time.time()
feat_time = end - start

# Create Brute-Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
start = time.time()
matches = bf.match(des1, des2)
end = time.time()
comp_time = end - start

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Print the features in each image
print("Features in Image 1: ", len(kp1))
print("Features in Image 2: ", len(kp2))

# Print the number of matched features
print("No. matched features: ", len(matches))

# Apply RANSAC to filter out outliers

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Select only the inliers

matches_mask = mask.ravel().tolist()

good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

good = len(good_matches)
# Print the number of matched features after removing outliers and finding the best match
print("No. of matched features after removing outliers and finding the best match: ", len(good_matches))

# Draw the matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.title('ORB'), plt.imshow(img3), plt.show()

# Display the matches
# cv2.imshow('Matches', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print feature description computation time
print("Feature description computation time (2 Images): ", feat_time)

# Print feature matching computation time
print("Feature matching computation time: ", match_time)

data['Algo'].append('ORB')
data['Features in Image 1'].append(len(kp1))
data['Features in Image 2'].append(len(kp2))
data['No. of Matched Features'].append(good)
data['No. of Matched Features after RANSAC'].append(n_inliers)
data['Feature Description Computation Time (2 Images) -'].append(comp_time)
data['Feature Matching Computation Time'].append(match_time)





# print('Features in Image 1 - ',len(kp1))
# print('Features in Image 2 - ',len(kp2))
# print('No. of Matched Features - ', len(good))
# print('No. of Matched Features after RANSAC - ', n_inliers)
# print('Feature Description Computation Time (2 Images) - ',comp_time)
# print('Feature Matching Computation Time - ',match_time)

# orb_detector = cv.ORB_create()

# # find the keypoints and descriptors with SIFT
# comp_start = time.time()
# kp1, des1 = orb_detector.detectAndCompute(img1,None)
# kp2, des2 = orb_detector.detectAndCompute(img2,None)
# comp_end = time.time()
# comp_time = comp_end - comp_start

# # BFMatcher with default params
# #ToDo - Can also try other matching algorithms - FLANN

# orb_surf = cv.BFMatcher()
# match_start = time.time()
# matches = orb_surf.knnMatch(des1,des2,k=2)
# match_end = time.time()
# match_time = match_end - match_start
# total_matches = len(matches)

# # Lowe's Ratio test
# good = []
# for m, n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# # Apply RANSAC to find the best set of matches
# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

# best_src_pts = None
# best_dst_pts = None
# best_inliers = 0

# # Ransac
# model, inliers = ransac(
#         (src_pts, dst_pts),
#         AffineTransform, min_samples=4,
#         residual_threshold=8, max_trials=10000
#     )

# n_inliers = np.sum(inliers)
# inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
# inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
# placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
# image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

# # cv2.imshow('Matches', image3)
# # cv2.waitKey(0)

# src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
# dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

# print('Features in Image 1 - ',len(kp1))
# print('Features in Image 2 - ',len(kp2))
# print('No. of Matched Features - ', len(good))
# print('No. of Matched Features after RANSAC - ', n_inliers)
# print('Feature Description Computation Time (2 Images) - ',comp_time)
# print('Feature Matching Computation Time - ',match_time)

akaze_detector = cv.AKAZE_create()

# find the keypoints and descriptors with SIFT
comp_start = time.time()
kp1, des1 = akaze_detector.detectAndCompute(img1,None)
kp2, des2 = akaze_detector.detectAndCompute(img2,None)
comp_end = time.time()
comp_time = comp_end - comp_start

# BFMatcher with default params
#ToDo - Can also try other matching algorithms - FLANN

akaze_surf = cv.BFMatcher()
match_start = time.time()
matches = akaze_surf.knnMatch(des1,des2,k=2)
match_end = time.time()
match_time = match_end - match_start
total_matches = len(matches)

# Lowe's Ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# Apply RANSAC to find the best set of matches
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

best_src_pts = None
best_dst_pts = None
best_inliers = 0

# Ransac
model, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform, min_samples=4,
        residual_threshold=8, max_trials=10000
    )

n_inliers = np.sum(inliers)
inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
plt.title('AKAZE'), plt.imshow(image3), plt.show()
# cv2.imshow('Matches', image3)
# cv2.waitKey(0)

src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

data['Algo'].append('AKAZE')
data['Features in Image 1'].append(len(kp1))
data['Features in Image 2'].append(len(kp2))
data['No. of Matched Features'].append(len(good))
data['No. of Matched Features after RANSAC'].append(n_inliers)
data['Feature Description Computation Time (2 Images) -'].append(comp_time)
data['Feature Matching Computation Time'].append(match_time)

# print('Features in Image 1 - ',len(kp1))
# print('Features in Image 2 - ',len(kp2))
# print('No. of Matched Features - ', len(good))
# print('No. of Matched Features after RANSAC - ', n_inliers)
# print('Feature Description Computation Time (2 Images) - ',comp_time)
# print('Feature Matching Computation Time - ',match_time)


df = pd.DataFrame(data)
print(df)


xAxis = ['SIFT','SURF','ORB','AKAZE']
yAxis1 = data['Features in Image 1']
yAxis2 = data['Features in Image 2']
yAxis3 = data['No. of Matched Features']
yAxis4 = data['No. of Matched Features after RANSAC']
yAxis5 = data['Feature Description Computation Time (2 Images) -']
yAxis6 = data['Feature Matching Computation Time']

plt.figure()
plt.plot(xAxis, yAxis1)
plt.title('Features Extracted in Img 1')
plt.xlabel('Feature Matching Techniques') 
plt.ylabel('# Features')
plt.show()

plt.figure()
plt.plot(xAxis, yAxis2)
plt.title('# Features Extracted in Img 2')
plt.xlabel('Feature Matching Techniques')
plt.ylabel('# Features')
plt.show()

plt.figure()
plt.plot(xAxis, yAxis3)
plt.title('# Matched Features in Image 1&2')
plt.xlabel('Feature Matching Techniques')
plt.ylabel('# Features')
plt.show()

plt.figure()
plt.plot(xAxis, yAxis4)
plt.title('# Matched Features after removing Ouliers')
plt.xlabel('Feature Matching Techniques')
plt.ylabel('# Features')
plt.show()

plt.figure()
plt.plot(xAxis, yAxis5)
plt.title('Time Taken for Feature Description')
plt.xlabel('Feature Matching Techniques')
plt.ylabel('Computation Time (in ms)')
plt.show()

plt.figure()
plt.plot(xAxis, yAxis6)
plt.title('Time Taken for Feature Matching')
plt.xlabel('Feature Matching Techniques')
plt.ylabel('Computation Time (in ms)')
plt.show()


