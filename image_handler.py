from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('00e9a277-ca5e-4350-95ce-8b2918b69fb9___RS_HL 4667.JPG')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
print(edges.shape)
