# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
img_path = sys.argv[1]
# Load the image
imageRoot = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# load the image
image = cv2.imread(img_path, cv2.IMREAD_COLOR)[740:len(imageRoot[0] -100),940:len(imageRoot[1])-100]

kernel = np.ones((5,5),np.uint8)
#plot_image(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#plot_image(gray)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#plot_image(closing)
closing = (closing * -1) + np.max(closing)
#plot_image(closing)
closing[closing < 100] = 0
#plot_image(closing) # %%
_, thresh = cv2.threshold(closing, 50, 255, cv2.THRESH_BINARY)

# Sort contours by their area in descending order
thresh = np.array(thresh, np.uint8)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Create a blank mask to fill polygons
mask = np.zeros_like(image)

# Define the distance threshold to determine proximity for union
distance_threshold = 20

# Utility function to check proximity between two points
def is_within_distance(point1, point2):
    return np.linalg.norm(point1 - point2) <= distance_threshold

# Function to merge contours representing polygons
def merge_polygons(contour1, contour2):
    merged_contour = np.vstack((contour1, contour2))
    hull = cv2.convexHull(merged_contour)
    return hull
poly_vertices = []
# Iterate through each contour
for i in range(len(contours)):
    # Approximate polygonal curves from the contour
    epsilon = 0.01 * cv2.arcLength(contours[i], True)
    approx = cv2.approxPolyDP(contours[i], epsilon, True)
    
    # Check if contour represents a polygon with at most 4 sides

    # Union adjacent polygons based on proximity
    for j in range(i+1, len(contours)):
        epsilon = 0.01 * cv2.arcLength(contours[j], True)
        approx_j = cv2.approxPolyDP(contours[j], epsilon, True)
        
        # Check proximity and merge polygons
        for point in approx:
            if any(is_within_distance(point[0], p[0]) for p in approx_j):
                approx = merge_polygons(approx, approx_j)
                break
    poly_vertices.append(approx.tolist())
    # Draw the merged polygon on the mask
    cv2.fillPoly(mask, [approx], 255)

# Union adjacent polygons by dilating the mask
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# Apply the unioned mask to the original image
result = cv2.bitwise_or(image, dilated_mask)

print(json.dumps(poly_vertices))
