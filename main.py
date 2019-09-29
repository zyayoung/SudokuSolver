import cv2
import numpy as np
from box import Box
from keras.models import load_model
import pytesseract
mnist = load_model("model.h5")

from solver import solve

solutions = {}
last_grid = ""
last_grid_cnt = 0
stable_grid = ""

cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
# cap = cv2.VideoCapture("2.jpg")
while True:
    ret, frame = cap.read()
    im = frame.copy()
    im_preprocess = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_preprocess = cv2.GaussianBlur(im_preprocess, (11, 11), 0)

    im_preprocess = cv2.adaptiveThreshold(im_preprocess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    # kernel = np.ones((3,3),np.uint8)
    # im_preprocess = cv2.morphologyEx(im_preprocess, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(
        im_preprocess,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    max_box = None
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)
        if len(contour) != 4:
            continue
        box = Box(contour)
        if box.area > im.shape[0]*im.shape[1] / 512 and box.max_cos < 0.5:
            if max_box is None or box.area > max_box.area:
                max_box = box

    # grid = Grid(im_preprocess)
    # if grid.add(max_box):
    #     grid.re_construct()

    if max_box:

        src = np.array(max_box.corners, dtype="float32")
        dst = np.array([
                [0, 0],
                [0, 84*9 - 1],
                [84*9 - 1, 84*9 - 1],
                [84*9 - 1, 0]
            ],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        # im_preprocess = cv2.warpPerspective(im_preprocess, M, (84*9, 84*9))
        im = cv2.warpPerspective(im, M, (84*9, 84*9))
        canvas = np.zeros((84*9, 84*9), dtype=np.uint8)

        im_preprocess = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_preprocess = cv2.GaussianBlur(im_preprocess, (5, 5), 0)
        im_preprocess = cv2.adaptiveThreshold(im_preprocess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,15)

        cols = im_preprocess.mean(axis=0)<127
        im_preprocess = im_preprocess[im_preprocess.mean(axis=1)<127,:]
        im_preprocess = im_preprocess[:,cols]
        
        kernel = np.array([
            [0,0,0],
            [1,1,1],
            [0,0,0],
        ], dtype=np.uint8)
        im_preprocess = cv2.morphologyEx(im_preprocess, cv2.MORPH_OPEN, kernel)
        kernel = np.array([
            [0,1,0],
            [0,1,0],
            [0,1,0],
        ], dtype=np.uint8)
        im_preprocess = cv2.morphologyEx(im_preprocess, cv2.MORPH_OPEN, kernel)
        # print(pytesseract.image_to_boxes(255-im_preprocess, config="--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"))
        im_preprocess = cv2.resize(im_preprocess, (36*9, 36*9), interpolation=cv2.INTER_LINEAR)
        grid = np.zeros((9,9), dtype=np.uint8)
        to_pred = []
        idx = []

        for i in range(9):
            for j in range(9):
                dig = im_preprocess[36*i+4:-4+36*(i+1),36*j+4:-4+36*(j+1)]
                if dig.mean()>4:
                    cols = dig.mean(axis=0)>1
                    dig = dig[dig.mean(axis=1)>1,:]
                    dig = dig[:,cols]
                    dig = cv2.resize(dig, (28, 28), interpolation=cv2.INTER_LINEAR)
                    dig = dig>127
                    to_pred.append(dig.reshape(28,28,1))
                    idx.append([i, j])
        to_pred = np.array(to_pred)
        if not to_pred.shape[0] > 0:
            cv2.imshow("d", frame)
            cv2.waitKey(1)
            continue
        idx = np.array(idx)
        pred = mnist.predict(to_pred)
        pred[:,0]=0
        pred = pred.argmax(axis=1)
        grid[idx.T[0], idx.T[1]] = pred
        # print(grid)
        
        grid_str = ''.join([str(i) for i in grid.flatten()]).replace('0', '.')
        if grid_str == last_grid:
            last_grid_cnt += 1
        else:
            last_grid_cnt = 0
            last_grid = grid_str
        if last_grid_cnt > 2:
            stable_grid = last_grid
        try:
            if stable_grid in solutions:
                assert solutions[stable_grid] != ""
                grid_solved = solutions[stable_grid]
            else:
                grid_solved = solve(stable_grid)
                print(grid_solved)
                solutions[stable_grid] = grid_solved
        except:
            solutions[stable_grid] = ""
            cv2.imshow("d", frame)
            cv2.waitKey(1)
            continue
        grid_solved = np.array([int(i) for i in grid_solved], dtype=int).reshape(9,9)

        for i in range(9):
            for j in range(9):
                if grid[i,j] == 0:
                    cv2.putText(canvas, str(grid_solved[i][j]), (84*j+32, 84*(i+1)-12), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=3)
        canvas = cv2.warpPerspective(canvas, M_inv, (frame.shape[1], frame.shape[0]))
        frame[canvas>0] = (255, 0, 0)
        cv2.imshow("d", frame)

        # _, contours, hierarchy = cv2.findContours(
        #     im_preprocess,
        #     cv2.RETR_LIST,
        #     cv2.CHAIN_APPROX_SIMPLE
        # )

        # im = cv2.cvtColor(im_preprocess, cv2.COLOR_GRAY2BGR)
        # for contour in contours:
        #     contourPerimeter = cv2.arcLength(contour, True)
        #     hull = cv2.convexHull(contour)
        #     contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)
        #     # if len(contour) > 4:
        #     #     continue
        #     cv2.drawContours(im, [contour], 0, (0, 0, 255), 2)

        # lines = cv2.HoughLines(im_preprocess, 1, np.pi / 180, 100, None, 0, 0)
    
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         eps = 0.1
        #         if not np.pi/2-eps < theta < np.pi/2+eps and not (theta<eps or theta>np.pi-eps):
        #             continue
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv2.line(im_preprocess, pt1, pt2, (0,0,0), 3, cv2.LINE_AA)
        # # kernel = np.array([
        # #     [0,1,0],
        # #     [1,1,1],
        # #     [0,1,0],
        # # ], dtype=np.uint8)
        # # im_preprocess = cv2.morphologyEx(im_preprocess, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("Demo", im_preprocess)
    cv2.waitKey(1)
