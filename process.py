import numpy as np
import cv2

test_size = (int(4896/4), int(3264/4))

img1 = cv2.resize(cv2.imread('/home/lucerna/Downloads/expo1_5/DSCF1892-2.jpg'), test_size) 
img2 = cv2.resize(cv2.imread('/home/lucerna/Downloads/expo1_5/DSCF1913-2.jpg'), test_size)

def detectStars(img):

    img_grey = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img_grey)
    return keypoints

keypoints = detectStars(img1)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img1, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('img1', im_with_keypoints)

keypoints = detectStars(img2)
im_with_keypoints = cv2.drawKeypoints(img2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img2', im_with_keypoints)
cv2.waitKey(0)