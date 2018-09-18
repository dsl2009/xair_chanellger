import cv2
import numpy as np
from matplotlib import pyplot as plt
rgb_image = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step2_train/番茄/番茄叶霉病/0a7b53c996273de373547e2e0b159c64.jpg')
w, h, c = rgb_image.shape
hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HLS)

rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
lower_red = np.array([26, 0, 0])
upper_red = np.array([99, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
plt.imshow(mask)
plt.show()
_, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

d = []
for ix, x in enumerate(contours):
    d.append(cv2.contourArea(x))
k = np.argmax(d)
print(k)
cv2.drawContours(rgb_image, contours, k, (0,255,0), -1)

rows,cols = rgb_image.shape[:2]
[vx,vy,x,y] = cv2.fitLine(contours[k], cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(rgb_image,(cols-1,righty),(0,lefty),(0,255,0),2)
plt.imshow(rgb_image)
plt.show()