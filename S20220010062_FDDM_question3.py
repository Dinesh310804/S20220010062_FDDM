import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_matching(image_path1, image_path2):
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Check if the images were loaded successfully
    if img1 is None or img2 is None:
        print("Error: Could not load one of the images.")
        return

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find matches
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        # Calculate SSD distance (which is the same as L2 distance in this context)
        ssd_distance1 = m.distance
        ssd_distance2 = n.distance

        # Calculate ratio distance
        if ssd_distance1 < 0.75 * ssd_distance2:
            good_matches.append(m)

    # Draw matches
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches with SIFT')
    plt.axis('off')  # Turn off axis labels
    plt.show()

    # Print details of the closest and second closest matches
    print("Number of good matches:", len(good_matches))
    if good_matches:
        print("Closest match SSD distance:", good_matches[0].distance)
        if len(good_matches) > 1:
            print("Second closest match SSD distance:", good_matches[1].distance)
            ratio_distance = good_matches[0].distance / good_matches[1].distance
            print("Ratio distance (closest/second closest):", ratio_distance)

# Paths to your two .png images
image_path_1 = "img2.png"
image_path_2 = "img4.png"

# Perform feature matching
sift_feature_matching(image_path_1, image_path_2)
