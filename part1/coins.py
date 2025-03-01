import numpy as np
import cv2
import os
import shutil

# Load the image and resize to 800x800
img = cv2.imread("input/coins_2.jpeg")
img = cv2.resize(img, (800, 800))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

#Use canny edge detector to find edges in an image
edges = cv2.Canny(blurred,50,150)

kernel = np.ones((5, 5), np.uint8)
morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # Keep contours that are roughly circular and within a reasonable area range
    if area > 500 :
        valid_contours.append(cnt)

# Filter contours based on area and circularity

if os.path.exists("output"):
    shutil.rmtree("output")

os.makedirs("output")

# Draw only the valid contours
output_image = img.copy()
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 2)

#Segmented image
segmented_image = np.zeros_like(img)
cv2.drawContours(segmented_image, valid_contours, -1, (0, 0, 255), 2)

# Create a mask to isolate the areas inside the contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
cv2.imwrite("output/Segmented_mask.png",mask)

segmented_image[mask == 255] = img[mask == 255]
cv2.putText(segmented_image, f"Coins: {len(valid_contours)}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("output/Segmented_image.png",segmented_image)


# Each segmented image :
coin_count = 0
for i, cnt in enumerate(valid_contours):
    # Create a mask for each coin (with a circular shape)
    coin_mask = np.zeros_like(gray)
    cv2.drawContours(coin_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Apply the coin mask to extract the coin's exact circular shape from the original image
    coin_image = cv2.bitwise_and(img, img, mask=coin_mask)
    
    # Find the bounding box of the contour to crop the circular region
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Crop the region containing the coin
    coin_cropped = coin_image[y:y+h, x:x+w]
    
    # Save the cropped coin image
    coin_count += 1
    coin_filename = f"output/segement-coin_{coin_count}.png"
    cv2.imwrite(coin_filename, coin_cropped)


# Print the detected number of coins
print("Number of coins detected:", len(valid_contours))

#Save the final output image
cv2.imwrite("output/detected_coins.png", output_image)

