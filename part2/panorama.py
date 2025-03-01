import numpy as np
import cv2
import os
import shutil

#Load the images and put it into an array
img1 = cv2.imread("input/img1.jpeg")
img2 = cv2.imread("input/img2.jpeg")
img3 = cv2.imread("input/img3.jpeg")
images = [img1,img2,img3]

# Ensure that at least 2 images are loaded for stitching
if len(images) < 2:
    print("Error: At least 2 images are required for stitching.")
else:

    if os.path.exists("output"):
        shutil.rmtree("output")

    os.makedirs("output")

    # Create SIFT detector
    sift = cv2.SIFT_create()

    #use BF matcher to matck keypoints
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    for idx, img in enumerate(images):
        kp1, des1 = sift.detectAndCompute(img, None)
        first_image = cv2.drawKeypoints(img, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        filename = f"output/image_{idx+1}.png"
        cv2.imwrite(filename, first_image)

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        filename = f"output/matches_{i+1}_{i+2}.png"
        cv2.imwrite(filename, img_matches)

    # Create a Stitcher object
    imageStitcher = cv2.Stitcher_create()
    error, stitched_img = imageStitcher.stitch(images)

    cv2.imwrite('output/output_without_descriptor.png', stitched_img)

    if error == cv2.Stitcher_OK:
        print("Stitching successful.")
        # Convert the stitched image to grayscale
        gray_stitched = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        keypoints_stitched, descriptors_stitched = sift.detectAndCompute(gray_stitched, None)

        # Draw keypoints on the stitched image
        stitched_with_keypoints = cv2.drawKeypoints(stitched_img, keypoints_stitched, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # Save the output image
        cv2.imwrite('output/output_descriptor.png', stitched_with_keypoints)
    else:
        print(f"Error during stitching. Code: {error}")
