#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import time
from PIL import Image, ImageDraw, ImageFont
run_img = cv.imread('run.jpg', 1)       # the run image about recording
stop_img = cv.imread('stop.jpg', 1)     # the stop image about recording
bighead = cv.imread('bighead.png', 1)   # my big head picture
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
cnt_video = 0                           # count the number of output videos
font = cv.FONT_HERSHEY_SIMPLEX
recordFlag = False                      # tell if the record func work
drawing = False                         # tell if the draw_line func work
ix, iy = -1, -1
iix, iiy = -1, -1
line_list = []                          # record the lines have been drawn
firstFlag = True                        # tell if the line is the first line


def img_resize(img, length, height):    # resize the image
    res = cv.resize(img, (length, height), interpolation=cv.INTER_CUBIC)
    return res


def draw_line(event, x, y, flags, param):
    global ix, iy, iix, iiy, drawing, line_list
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        tempFlag = True
        ix, iy = x, y
        iix, iiy = ix, iy
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            tempFlag = False
            iix, iiy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        line_list.append([ix, iy, iix, iiy])


def paint_chinese_opencv(img, chinese, pos, color):
    img_PIL = Image.fromarray(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    chinese_font = ImageFont.truetype('simhei.ttf', 15, encoding='utf-8')
    position = pos
    fillColor = color
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=chinese_font, fill=fillColor)
    img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)
    return img


run_img = img_resize(run_img, 30, 30)
stop_img = img_resize(stop_img, 30, 30)
bighead = img_resize(bighead, 60, 60)
cv.namedWindow('frame')
cv.setMouseCallback('frame', draw_line)
out = -1                                # initialize the output video
while 1:
    ret, frame = cap.read()
    if recordFlag:                      # output the record state
        frame[440:470, 10:40] = run_img
    else:
        frame[440:470, 10:40] = stop_img
    frame[10:70, 580:640] = bighead
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv.putText(frame, localtime, (50, 470), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    frame = paint_chinese_opencv(frame, "徐骏凯 21960439", (250, 460), (255, 255, 255))
    for line in line_list:              # draw all the lines have been drawn
        cv.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)
    cv.line(frame, (ix, iy), (iix, iiy), (0, 0, 255), 1)  # draw current line
    if recordFlag:
        out.write(frame)                # if recordFlag is True, record the frame in the current out file
    cv.imshow('frame', frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):                   # enter 'q' to quit
        break
    elif k == ord(' '):                 # enter ' ' to start record or quit record
        if recordFlag:
            out.release()
            recordFlag = False
            cnt_video += 1
        else:
            recordFlag = True
            out = cv.VideoWriter('output%d.avi' % cnt_video, fourcc, 20.0, (640, 480))
    elif k == ord('c'):                 # enter 'c' to clear all the lines
        line_list.clear()
        ix, iy = -1, -1
        iix, iiy = -1, -1

cap.release()
cv.destroyAllWindows()
