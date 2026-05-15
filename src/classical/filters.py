import cv2
import numpy as np

# -----------------------------------
# Sobel Edge Extraction
# -----------------------------------

def sobel_edges(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    sobelx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    sobely = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1
    )

    magnitude = np.sqrt(
        sobelx**2 + sobely**2
    )

    magnitude = np.clip(
        magnitude,
        0,
        255
    )

    magnitude = magnitude.astype(
        np.uint8
    )

    magnitude = cv2.cvtColor(
        magnitude,
        cv2.COLOR_GRAY2RGB
    )

    return magnitude