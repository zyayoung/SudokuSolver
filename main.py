import cv2
import numpy as np
import time
from box import Box
from solver import solve

net = cv2.dnn.readNetFromTensorflow("d.pb")


solutions = {}
last_grid = ""
last_grid_cnt = 0
stable_grid = ""

cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
video_writer = cv2.VideoWriter()
video_writer.open("out.mp4", cv2.VideoWriter_fourcc('X','2','6','4'), 12, (1280,720))

def get_max_box(im):
    im_preprocess = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_preprocess = cv2.GaussianBlur(im_preprocess, (11, 11), 0)

    im_preprocess = cv2.adaptiveThreshold(im_preprocess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    _, contours, _ = cv2.findContours(
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
    return max_box

def remove_lines(im):
    cols = im.mean(axis=0)<127
    im = im[im.mean(axis=1)<127,:]
    im = im[:,cols]
    kernel = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0],
    ], dtype=np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    kernel = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0],
    ], dtype=np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    return im

def grid2str(grid):
    return ''.join([str(i) for i in grid.flatten()]).replace('0', '.')

def str2grid(grid):
    return np.array([int(i) for i in grid], dtype=int).reshape(9,9)
    
frame = None

while True:
    if frame is not None:
        cv2.imshow("d", frame)
        video_writer.write(frame)
        cv2.waitKey(1)

    ret, frame = cap.read()
    if not ret:
        video_writer.release()
        break

    max_box=get_max_box(frame)
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

        im_preprocess = cv2.warpPerspective(frame, M, (84*9, 84*9))
        canvas = np.zeros((84*9, 84*9), dtype=np.uint8)

        im_preprocess = cv2.cvtColor(im_preprocess, cv2.COLOR_BGR2GRAY)
        im_preprocess = cv2.GaussianBlur(im_preprocess, (5, 5), 0)
        im_preprocess = cv2.adaptiveThreshold(im_preprocess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,15)

        im_preprocess = remove_lines(im_preprocess)
        im_preprocess = cv2.resize(im_preprocess, (36*9, 36*9), interpolation=cv2.INTER_LINEAR)
        grid = np.zeros((9,9), dtype=np.uint8)

        to_pred = []
        to_pred_idx = []
        for i in range(9):
            for j in range(9):
                dig = im_preprocess[36*i+4:-4+36*(i+1),36*j+4:-4+36*(j+1)]
                # if something in dig
                if dig.mean()>4:
                    # remove empty rows and cols in dig
                    cols = dig.mean(axis=0)>1
                    dig = dig[dig.mean(axis=1)>1,:]
                    dig = dig[:,cols]

                    # prepare for prediction
                    dig = cv2.resize(dig, (28, 28), interpolation=cv2.INTER_LINEAR)
                    to_pred.append(dig.reshape(28,28,1))
                    to_pred_idx.append([i, j])
        to_pred = np.array(to_pred, dtype=np.float32)
        to_pred_idx = np.array(to_pred_idx)
        if not to_pred.shape[0] > 0: continue
        
        # Predict
        blob = cv2.dnn.blobFromImages(to_pred/255)
        net.setInput(blob)
        pred = net.forward()
        pred[:,0]=0
        pred = pred.argmax(axis=1)

        # Apply prediction
        grid[to_pred_idx.T[0], to_pred_idx.T[1]] = pred
        cur_grid = grid2str(grid)

        if cur_grid == last_grid:
            last_grid_cnt += 1
        else:
            last_grid_cnt = 0
            last_grid = cur_grid
        
        # Update stable grid
        if last_grid_cnt > 2:
            stable_grid = last_grid
        
        # Skip if stable grid differ a lot from current grid
        if stable_grid and (np.array(list(cur_grid)) != np.array(list(stable_grid))).sum() > 4:
            continue

        try:
            if stable_grid in solutions:
                assert solutions[stable_grid] != ""
                grid_solved = solutions[stable_grid]
            else:
                grid_solved = solve(stable_grid)
                solutions[stable_grid] = grid_solved
        except:
            solutions[stable_grid] = ""
            continue

        grid_solved = str2grid(grid_solved)

        for i in range(9):
            for j in range(9):
                if grid[i,j] == 0:
                    cv2.putText(canvas, str(grid_solved[i][j]), (83*j+28, 83*(i+1)-14), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=3)
        canvas = cv2.warpPerspective(canvas, M_inv, (frame.shape[1], frame.shape[0]))
        frame[canvas>0] = (255, 0, 0)
