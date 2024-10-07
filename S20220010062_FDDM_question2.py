import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_detection(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return img_with_keypoints, descriptors  # Return the image with keypoints and descriptors

# Paths to your two .png images
image_path_1 = "img2.png"
image_path_2 = "img4.png"

# Perform SIFT feature detection on both images
result_img1, descriptors1 = sift_feature_detection(image_path_1)
result_img2, descriptors2 = sift_feature_detection(image_path_2)

# Display both images in a single output
plt.figure(figsize=(10, 5))

# Display first image with keypoints
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints - Image 1')
plt.axis('off')  # Turn off axis labels

# Display second image with keypoints
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints - Image 2')
plt.axis('off')  # Turn off axis labels

# Show the combined image
plt.tight_layout()
plt.show()

# Print out the shape of the descriptors
print(f'Descriptors for Image 1: {descriptors1.shape if descriptors1 is not None else "No descriptors found"}')
print(f'Descriptors for Image 2: {descriptors2.shape if descriptors2 is not None else "No descriptors found"}')
