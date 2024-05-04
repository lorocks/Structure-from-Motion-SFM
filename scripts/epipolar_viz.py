import numpy as np
import cv2

# Set of functions to calculate and display the epipoles and epipolar lines

def getMinMaxXY(x_min,x_max,y_min,y_max,line_info):
    if x_min == 0:
        y_min = y_max = -line_info[2]/line_info[1]
    else:
        x_min = -((line_info[1] * y_min) + line_info[2])/line_info[0]
        x_max = -((line_info[1] * y_max) + line_info[2])/line_info[0]
    return x_min, x_max, y_min, y_max

def drawEpipoles(points,F,image):
    draw_image = image.copy()
    for ppoint in points:
        point = np.array([[ppoint[0]], [ppoint[1]], [1]])
        line = np.dot(F, point)
        x_min, x_max, y_min, y_max = getMinMaxXY(None, None, 0, image.shape[0], line)
        cv2.circle(draw_image, (int(point[0][0]),int(point[1][0])), 10, (0,0,255), -1)
        cv2.line(draw_image, (int(x_min[0]), int(y_min)), (int(x_max[0]), int(y_max)), (0, 255, 0), 2)
    return draw_image

def drawRectEpipoles(points,F,image):
    draw_image = image.copy()
    for ppoint in points:
        point = np.array([[ppoint[0]],[ppoint[1]], [1]])
        line = np.dot(F, point)
        x_min, x_max, y_min, y_max = getMinMaxXY(0, image.shape[1] - 1, None, None, line)
        cv2.circle(draw_image, (int(point[0][0]),int(point[1][0])), 10, (0,0,255), -1)
        cv2.line(draw_image, (int(x_min), int(y_min[0])), (int(x_max), int(y_max[0])), (0, 255, 0), 2)
    return draw_image