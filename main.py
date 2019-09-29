import cv2
import numpy as np
from box import Box

im = cv2.imread("demo.jpg")

im_preprocess = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_preprocess = cv2.GaussianBlur(im_preprocess, (11, 11), 0)

im_preprocess = cv2.adaptiveThreshold(im_preprocess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

_, contours, hierarchy = cv2.findContours(
    im_preprocess,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

boxes = []
corners = []

for contour in contours:
    contourPerimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)
    if len(contour) != 4:
        continue
    box = Box(contour)
    if box.area > im.shape[0]*im.shape[1] / 256 and box.max_cos < 0.26:
        # cv2.drawContours(im, [contour], 0, (0, 0, 255), 2)
        boxes.append(box)
        corners.extend(box.corners)

corners = np.array(corners)
for box in boxes:
    near_cnt = np.array([(np.sqrt(np.square(corners-corner).sum(axis=1))<32).sum()-1 for corner in box.corners])
    if near_cnt.min() >= 3:
        print(box.center)

# cv2.imshow("Demo", im)
# cv2.waitKey(0)
