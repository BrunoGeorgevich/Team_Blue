#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:25:14 2019

@author: bruno
"""

import cv2
import cv2.aruco as aruco

from time import sleep

from random import randint, random

import paho.mqtt.client as mqtt_client

import tensorflow as tf
from object_detection.utils import label_map_util

from minimax import ai_turn, clean, wins, render, empty_cells, game_over

import numpy as np

calibrationFile = "calibration_rgb.yml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("camera_matrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

HUMAN = 1
COMP = -1

h_choice = 'O'
c_choice = 'X'

PATH_TO_CKPT = 'graph.pb'
PATH_TO_LABELS = 'labelmap.pbtxt'
NUM_CLASSES = 2

CLIENT_ID = 'user'
BROKER_ADDRESS = '10.0.0.105'

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
markerLength = 0.09

def HSV_color_thresholding(image, min_val_hsv, max_val_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_val_hsv, max_val_hsv)
    imask = mask > 0
    extracted = np.zeros_like(image, image.dtype)
    extracted[imask] = image[imask]
    return extracted

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
    global CENTERS
    frame = frame.copy()

    i = 0
    j = 1
    for row in CENTERS:
        for cell in row:
            p1, p2 = dot(cell,3)
            if i == 0:
                frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 5*j)
            elif i == 1:
                frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 5*j)
            elif i == 2:
                frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 5*j)
            j += 1
        i += 1
        j = 1
    return frame

def retrieve_central_dots(frame):
    frame = frame.copy()

    rows = [
            HSV_color_thresholding(frame, (74,126,180), (136,255,255)),
            HSV_color_thresholding(frame, (46,49,0), (77,255,255)),
            HSV_color_thresholding(frame, (0,93,0), (29,255,255))
            ]

    dots = []

    for row in rows:
        gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        aux_dots = []
    
        frame_area = gray.shape[0] * gray.shape[1]
    
        for c in cnts:
    
            cnt_area = cv2.contourArea(c)
            if cnt_area < 0.000001*frame_area:
                continue
    
            x,y,w,h = cv2.boundingRect(c)
            aux_dots.append((x,y,w,h,cnt_area))
    
        aux_dots.sort(key=lambda tup: tup[4])
        aux_dots = list(map(lambda tup: (tup[0], tup[1]), aux_dots))
        dots.append(aux_dots)

    return frame, dots

def retrieve_cells(distorced_warped):
    global W, H, CENTERS

    cells = {}

    idx = 1

    for row in CENTERS:
        for cell in row:
            x1, x2 = (-W/2 + cell[0], W/2 + cell[0])
            y1, y2 = (-H/2 + cell[1], H/2 + cell[1])
            cells[idx] = distorced_warped[int(y1):int(y2), int(x1):int(x2)]
            idx += 1
    return cells

def predict(image):
    global sess

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    predictions = list(zip(np.squeeze(boxes),
    	 np.squeeze(classes).astype(np.int32),
    	 np.squeeze(scores)))

    predictions.sort(key=lambda it: it[2], reverse=True)
    return predictions[0]

def draw_prediction(image, prediction):
    global category_index

    image = image.copy()
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h,w,d = image.shape
    
    x1,y1,x2,y2 = prediction[0] * [w,h,w,h]

    score = prediction[2]
    class_name = category_index[prediction[1]]['name']

    if score < 0.8:
        class_name = "-"

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
    cv2.putText(image, f"{class_name} | {score:.2f}", (int((x2 - x1)/6), int((y2 - y1)/6)), cv2.FONT_HERSHEY_SIMPLEX, int(h*0.009),(0,200,0), int(w*0.009))
    return image

def randomize_image(image):
    noised = np.zeros_like(image)
    channels = cv2.split(noised)
    
    for channel in channels:
        channel = cv2.randn(channel, 128, 64)
    
    noised = cv2.merge(channels)
    
    foreground = np.zeros_like(image)
    channels = cv2.split(foreground)
    
    for channel in channels:
        channel +=  np.uint8(30 + 200*random())
    
    foreground = cv2.merge(channels)
    
    image[image == 255] = noised[image == 255]
    image[image == 0] = foreground[image == 0]

    return image

def current_board_state(cells):
    board = '''
        | 1 | 2 | 3 |
        | 4 | 5 | 6 |
        | 7 | 8 | 9 |
    '''

    for i,k in enumerate(cells.keys()):
        cell = cells[k]

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        aux = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    75,
                                    3)

        cnts, _ = cv2.findContours(aux, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = list(map(lambda it: (it, cv2.contourArea(it)), cnts))
        cnts = list(filter(lambda it: it[1] > 250, cnts))
        cnts.sort(key=lambda it: it[1], reverse=True)

        if len(cnts) <= 0:
            board = board.replace(f'{k}', ' ')
            continue

        cnts = cnts[0][0]

        bg = np.ones_like(aux)*255
        x,y,w,h = cv2.boundingRect(cnts)
        bg[y:y+h,x:x+w] = ~aux[y:y+h,x:x+w]

        gray = cv2.merge([bg, bg, bg])
        gray = randomize_image(gray)

        cv2.imshow(str(k), gray)

        prediction = predict(gray)

        score = prediction[2]
        class_name = category_index[prediction[1]]['name']

        # print(f'{k} -> SCORE {score:.2f} : {class_name}')

        if score < 0.98:
            class_name = " "

        board = board.replace(f'{k}', class_name)

    def remap(class_name):
        if class_name == 'X':
            return -1
        elif class_name == 'O':
            return 1
        elif class_name == '':
            return 0

    board_symbol = board.replace(' ','').replace('|\n','').replace('\n|','').split("|")
    board_symbol = list(map(remap, board_symbol))

    # print(board)
    return board_symbol

def get_board_state(distorced_pts):
    M = cv2.getPerspectiveTransform(distorced_pts,reference_pts)
            
    rows,cols,ch = reference.shape
    distorced_warped = cv2.warpPerspective(distorced,M,(cols,rows))
    distorced_warped_dotted = draw_dots(distorced_warped)

    # cv2.imshow("Distorced Warped", distorced_warped)

    cells = retrieve_cells(distorced_warped)
    board_symbol = current_board_state(cells)
    
    M = cv2.getPerspectiveTransform(reference_pts, distorced_pts)

    rows,cols,ch = distorced.shape
    distorced_reverse_warped = cv2.warpPerspective(distorced_warped_dotted,M,(cols,rows))
    distorced_reverse_warped, dots = retrieve_central_dots(distorced_reverse_warped)
    return dots, board_symbol, distorced_warped, distorced_reverse_warped

def check_victory(board):

    if wins(board, HUMAN):
        render(board, c_choice, h_choice)
        print('YOU WIN!')
    elif wins(board, COMP):
        render(board, c_choice, h_choice)
        print('YOU LOSE!')
    else:
        render(board, c_choice, h_choice)
        print('DRAW!')

def send_mqtt_message(message, topic):
    global CLIENT_ID, BROKER_ADDRESS
    client = mqtt_client.Client(CLIENT_ID)
    client.connect(BROKER_ADDRESS)
    client.publish(topic, message)
    client.disconnect()

def send_board_state_message(board):
    board_str = list(map(lambda it: str(it), board))
    send_mqtt_message(','.join(board_str), 'board_state')

def send_move_message(y,x):
    send_mqtt_message(f'{y},{x}', 'move')

reference = cv2.imread("assets/reference_board_2.png")

BOARD_W = reference.shape[1]
BOARD_H = reference.shape[0]

H = int(BOARD_H*0.11)
W = int(BOARD_W*0.11)

CENTERS = [[(int(BOARD_W*0.33),int(BOARD_H*0.35)), (int(BOARD_W*0.5),int(BOARD_H*0.35)), (int(BOARD_W*0.65),int(BOARD_H*0.35))],
           [(int(BOARD_W*0.33),int(BOARD_H*0.52)) , (int(BOARD_W*0.5),int(BOARD_H*0.5)) , (int(BOARD_W*0.65),int(BOARD_H*0.5)) ],
           [(int(BOARD_W*0.33),int(BOARD_H*0.66)), (int(BOARD_W*0.5),int(BOARD_H*0.66)), (int(BOARD_W*0.65),int(BOARD_H*0.66))]]

reference_aruco, reference_tags = detect_aruco(reference)

reference_pts = np.float32([
            list(reference_tags[0]['center']),
            list(reference_tags[1]['center']),
            list(reference_tags[2]['center']),
            list(reference_tags[3]['center'])
        ])

# cv2.namedWindow("Reference", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Distorced", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Reverse Warped", cv2.WINDOW_KEEPRATIO)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
current_turn = HUMAN

last_frame_board = [[None,None, None],[None,None, None],[None,None, None]]
equal_times = 0

while cv2.waitKey(1) != 27:
    _,distorced = cap.read()
    distorced = cv2.resize(distorced, (640, 480))
    distorced = cv2.cvtColor(distorced, cv2.COLOR_BGR2GRAY)
    distorced = cv2.merge([distorced, distorced, distorced])

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

                dots, board, distorced_warped, distorced_reverse_warped = get_board_state(distorced_pts)

                # cv2.imshow("Reverse Warped", distorced_warped)

                send_board_state_message(board)

                board = np.reshape(board, (3,3))

                print(f"ANALISANDO {equal_times + 1}/8!!")

                if equal_times < 8  :
                    if (last_frame_board == board).all():
                        equal_times += 1
                    else:
                        equal_times = 0
                    last_frame_board = board
                    continue

                if len(empty_cells(board)) <= 0 or game_over(board):
                    check_victory(board)
                    break

                clean()

                if current_turn == COMP:
                    print("VEZ DO COMPUTADOR!")
                    board,y,x = ai_turn(c_choice, h_choice, board)
                    send_move_message(x,y)
                elif current_turn == HUMAN:
                    print("VEZ DO HUMANO!")

                render(board, c_choice, h_choice)

                print("QUANDO FINALIZAR A JOGADA APERTE Q!")

                while cv2.waitKey(1) != ord("q"):
                    _,distorced = cap.read()
                    distorced = cv2.resize(distorced, (1280, 960))
                    distorced = cv2.cvtColor(distorced, cv2.COLOR_BGR2GRAY)
                    distorced = cv2.merge([distorced, distorced, distorced])
                    cv2.imshow("Distorced", distorced)

                equal_times = 0
                last_frame_board = board

                if current_turn == HUMAN:
                    current_turn = COMP
                elif current_turn == COMP:
                    current_turn = HUMAN

    else:
        print("NAO FORAM DETECTADAS AS 4 TAGS!")
    cv2.imshow("Distorced", distorced)

cap.release()
cv2.destroyAllWindows()
#%%