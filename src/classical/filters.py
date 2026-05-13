import cv2
import numpy as np

def gaussian_filter(image):

    return cv2.GaussianBlur(image, (3,3), 0)

def median_filter(image):

    return cv2.medianBlur(image, 3)

def sobel_filter(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sobelx**2 + sobely**2)

    return magnitude