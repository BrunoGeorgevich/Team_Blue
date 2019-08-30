#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:25:14 2019

@author: bruno
"""

import cv2
import cv2.aruco as aruco

from skimage.morphology import reconstruction

import numpy as np

calibrationFile = "calibration_rgb.yml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("camera_matrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
markerLength = 0.09

def detect_aruco(frame):
    frame = frame.copy()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    img_with_aruco  = frame

    tags = {}

    if ids is None:
        return img_with_aruco, None

    if len(ids) > 0:
        for i in range(len(ids)):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[i]], markerLength, camera_matrix, dist_coeffs)

            tags[ids[i][0]] = {}
            tags[ids[i][0]]['rvecs'] = rvecs
            tags[ids[i][0]]['tvecs'] = tvecs
            # img_with_aruco = cv2.aruco.drawDetectedMarkers(frame, [corners[i]], ids[i], (200, 50, 200))

            p1 = [999999999,999999999]
            p2 = [0,0]

            for c in corners[i][0]:
                if c[0] < p1[0]:
                    p1[0] = c[0]
                if c[1] < p1[1]:
                    p1[1] = c[1]
                if c[0] > p2[0]:
                    p2[0] = c[0]
                if c[1] > p2[1]:
                    p2[1] = c[1]

            p1 = tuple(p1)
            p2 = tuple(p2)

            center  = (int((p2[0] + p1[0])/2),int((p2[1] + p1[1])/2))
            center1 = (int((p2[0] + p1[0])/2 - 2),int((p2[1] + p1[1])/2 - 2))
            center2 = (int((p2[0] + p1[0])/2 + 2),int((p2[1] + p1[1])/2 + 2))

            tags[ids[i][0]]['p1'] = p1
            tags[ids[i][0]]['p2'] = p2
            tags[ids[i][0]]['center'] = center

            img_with_aruco = cv2.rectangle(img_with_aruco, p1, p2, (0, 0, 255), 5)
            img_with_aruco = cv2.rectangle(img_with_aruco, center1, center2, (0, 255, 0), 5)

            img_with_aruco = cv2.aruco.drawAxis(img_with_aruco, camera_matrix, dist_coeffs, rvecs, tvecs, markerLength)

    return img_with_aruco, tags

def dot(tup, space = 1):
    x = tup[0]
    y = tup[1]
    return (x - space, y - space), (x + space, y + space)

def draw_dots(frame):
    frame = frame.copy()
    centers = [[(256,184), (420,184), (588,184)],
               [(256,300), (420,300), (588,300)],
               [(256,416), (420,416), (588,416)]]

    for row in centers:
        for cell in row:
            p1, p2 = dot(cell,3)
            frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    return frame


reference = cv2.imread("assets/reference.png")

reference_aruco, reference_tags = detect_aruco(reference)
    
reference_pts = np.float32([
            list(reference_tags[0]['center']),
            list(reference_tags[1]['center']),
            list(reference_tags[2]['center']),
            list(reference_tags[3]['center'])
        ])

cv2.namedWindow("Reference", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Distorced", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Distorced Warped", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Reverse Warped", cv2.WINDOW_KEEPRATIO)

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord("q"):
    _,distorced = cap.read()
    distorced_aruco, distorced_tags = detect_aruco(distorced)

    if distorced_tags is not None and len(distorced_tags.keys()) == 4:
        if 0 in distorced_tags.keys() and \
           1 in distorced_tags.keys() and \
           2 in distorced_tags.keys() and \
           3 in distorced_tags.keys():
            distorced_pts = np.float32([
                        list(distorced_tags[0]['center']),
                        list(distorced_tags[1]['center']),
                        list(distorced_tags[2]['center']),
                        list(distorced_tags[3]['center'])
                    ])
            
            M = cv2.getPerspectiveTransform(distorced_pts,reference_pts)
            
            rows,cols,ch = reference.shape
            distorced_warped = draw_dots(cv2.warpPerspective(distorced,M,(cols,rows)))
            cv2.imshow("Distorced Warped", distorced_warped)
            
            M = cv2.getPerspectiveTransform(reference_pts, distorced_pts)
    
            rows,cols,ch = distorced.shape
            distorced_reverse_warped = cv2.warpPerspective(distorced_warped,M,(cols,rows))
            cv2.imshow("Reverse Warped", distorced_reverse_warped)

    cv2.imshow("Reference", draw_dots(reference))
    cv2.imshow("Distorced", distorced)

cap.release()
cv2.destroyAllWindows()
#%%