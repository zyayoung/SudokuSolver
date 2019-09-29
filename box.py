import cv2
import numpy as np

class Box:
    def __init__(self, contour):
        assert len(contour) == 4
        center = contour[:, 0].mean(axis=0)
        self.contour = np.array(sorted(contour, key=lambda x:np.arctan2(*(x[0]-center))))  # order: nw, sw, se, ne

    @property
    def area(self):
        return cv2.contourArea(self.contour)

    @property
    def corners(self):
        return self.contour[:, 0]

    @property
    def center(self):
        return self.corners.mean(axis=0)
    
    @property
    def max_cos(self):
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
            return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))
        
        contour = self.contour.reshape(-1, 2)
        max_cos = np.max(
            [angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)]
        )
        return max_cos
    
    def __str__(self):
        return self.contour.__str__()
