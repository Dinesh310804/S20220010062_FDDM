import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32
    gray = np.float32(gray)
    
    # Apply the Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Dilate the result to mark the corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value to detect strong corners
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    return img  # Return the image with corners marked

# Paths to your two .png images
image_path_1 = "img2.png"
image_path_2 = "img4.png"

# Perform Harris corner detection on both images
result_img1 = harris_corner_detection(image_path_1)
result_img2 = harris_corner_detection(image_path_2)

# Display both images in a single output
plt.figure(figsize=(10, 5))

# Display first image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners - Image 1')
plt.axis('off')  # Turn off axis labels

# Display second image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners - Image 2')
plt.axis('off')  # Turn off axis labels

# Show the combined image
plt.tight_layout()
plt.show()
