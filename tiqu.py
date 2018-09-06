import cv2
import numpy as np
from matplotlib import pyplot as plt
rgb_image = cv2.imread('0a85f197-fa0e-4665-be11-d52e84a45878___UF.GRC_YLCV_Lab 02732.JPG')
w, h, c = rgb_image.shape
hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HLS)
rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
lower_red = np.array([26, 0, 0])
upper_red = np.array([99, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
print(mask.shape)
print(mask)
print(type(mask))
_, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

d = []
for ix, x in enumerate(contours):
    d.append(cv2.contourArea(x))
k = np.argmax(d)
print(k)
cv2.drawContours(rgb_image, contours, k, (0,255,0), 3)
rows,cols = rgb_image.shape[:2]
[vx,vy,x,y] = cv2.fitLine(contours[k], cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(rgb_image,(cols-1,righty),(0,lefty),(0,255,0),2)
plt.imshow(rgb_image)
plt.show()