import numpy as np
import cv2

# Import picture from the local file named "IMG_7678.jpg"
img = cv2.imread("exercise0425/IMG_7678.jpg")
height, width, _ = img.shape  # Obtain the size of the image
"""
for point in [[1560,1728],[2304,1500],[1616, 2760],[2400,2808]]:
    cv2.circle(img, tuple(point), 10, (255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
# These code is to decide which pixels are the edge of the building. It shows the pixel which I mentioned in the line 8 by drawing circle on the input image
"""
cv2.imshow('input', img) # Show the image before the transformation
cv2.waitKey(0) # Pause the process until you press enter or esc key

pts1 = np.float32([[1560, 1728], [2304, 1500], [1616, 2760], [2400, 2808]]) # Input 4 points which is equal to the edge of the building
pts2 = np.float32([[500, 500], [2500, 500], [500, 3500], [2500, 3500]]) # Define 4 points which the points I mentioned above goes to.
M = cv2.getPerspectiveTransform(pts1, pts2) # Generate the matrix of projective transformation
dst = cv2.warpPerspective(img, M, (3000, 4000)) # Execute the transformation

cv2.imshow('output', dst) # Show the image after the transformation
cv2.waitKey(0) # Pause the process until you press enter or esc key

cv2.destroyAllWindows() # Close the window
