import cv2
import numpy as np

# Read the aerial image
image = cv2.imread("sample_inputs/aerial_original.png")

# Check whether image loaded correctly
if image is None:
    print("Could not load the image.")
    exit()

# Create a white layer
white = np.full_like(image, 255)

# Blend original image with white layer to simulate fog
foggy_image = cv2.addWeighted(
    image, 0.55,
    white, 0.45,
    0
)

# Save the foggy image
cv2.imwrite("sample_inputs/aerial_foggy.jpg", foggy_image)

print("Foggy image created successfully!")