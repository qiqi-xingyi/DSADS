# -*- coding: utf-8 -*-
# @Time    : 2021/4/19 22:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MainWindow.py
# @Software: PyCharm
import sys,cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
#窗口控件
from ui_MainWindow3 import Ui_MainWindow
#初始动画
from initial_animation import SplashScreen


import sys
import os
import cv2

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import shutil
from detect_one import PoseDect
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class Count_Line(object):
    def __init__(self, two_points):
        self.count_line = two_points
        self.car_count = 0


class Line(object):
    def __init__(self, line_points, box, label):
        self.line_points = line_points
        self.isCount = False
        self.Classname = label
        self.isClassify = False
        self.box = box
        self.velocity = 0
        self.velocity_x = 0

    def get_velocity(self):
        if (len(self.line_points) >= 2):
            self.velocity = (self.line_points[0][0] - self.line_points[0][1]) + (
                        self.line_points[1][0] - self.line_points[1][1])
            self.velocity = abs(self.velocity / 180) * 3.6
            self.velocity = int(self.velocity)

    def get_velocity_x(self):
        if (len(self.line_points) >= 2):
            self.velocity_x = (self.line_points[0][0] - self.line_points[-1][0])
            self.velocity_x = abs(self.velocity_x)
            self.velocity_x = round(self.velocity_x, 2)


class Car(object):
    def __init__(self):
        self.box_points = []
        self.all_points = []
        self.lines = []
        self.time_count = 0
        self.M = 15  # 长度
        self.N = 900  # 最小距离
        self.L = 10  # 时间
        self.findLgiht = False
        self.tafficLightState = 0
        self.tafficLightBox = None

    def update_point(self):
        if (self.box_points != None):
            if (len(self.lines) == 0):
                for points in self.all_points:
                    if (points[-1] == 'traffic light'):
                        self.tafficLightBox = points[0:4]
                        self.findLgiht = True
                        continue
                    head_points = [(points[0] + points[2]) / 2, points[3]]
                    # self.lines.append([0, [points[0], points[1], self.time_count]])
                    self.lines.append(Line([[head_points[0], head_points[1], self.time_count]], points[0:4],
                                           points[4]))  # create new line
            else:
                for points in self.all_points:
                    if (points[-1] == 'traffic light'):
                        self.tafficLightBox = points[0:4]
                        self.findLgiht = True
                        continue
                    head_points = [(points[0] + points[2]) / 2, points[3]]
                    mindes = 9999999
                    minnum = 9999999
                    line_count = 0
                    for line in self.lines:
                        des = (line.line_points[-1][0] - head_points[0]) ** 2 + (
                                    line.line_points[-1][1] - head_points[1]) ** 2
                        if (des < mindes):
                            mindes = des
                            minnum = line_count
                        line_count += 1  # line_count denote minest distance between two points

                    if (mindes < self.N):  # add new point into minest distance line
                        self.lines[minnum].line_points.append([head_points[0], head_points[1], self.time_count])
                    else:  # else create a new line
                        self.lines.append(
                            Line([[head_points[0], head_points[1], self.time_count]], points[0:4], points[4]))
            for line in self.lines:
                pass
                # print("name", line.Classname)
                # print("box", line.box)

    def delete_point(self):
        count_line = 0
        for line in self.lines:
            count_point = 0
            for point in line.line_points:
                if (self.time_count - point[2] >= self.L):
                    del (self.lines[count_line].line_points[count_point])
                count_point += 1

            if (len(self.lines[count_line].line_points) == 0):
                del (self.lines[count_line])
            elif (len(self.lines[count_line].line_points) >= self.M + 1):
                del (self.lines[count_line].line_points[0])  # important!!!!
            count_line += 1

    def draw_line(self, img):
        # cv2.putText(img, str(Return_Car[0]), (int(Recice_Line[0][0][0]) + 50, int(Recice_Line[0][0][1])),
        #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        # cv2.putText(img, "Passed Car:" + str(Return_Car[1]), (int(Recice_Line[1][0][0]) + 10, int(Recice_Line[1][0][1]) - 10),
        #             cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        # cv2.line(img, (int(Recice_Line[1][0][0]), int(Recice_Line[1][0][1])),(int(Recice_Line[1][1][0]), int(Recice_Line[1][1][1])), (0, 0, 255), 6)
        #
        # if self.findLgiht == True:
        #     img0 = img[self.tafficLightBox[1]:self.tafficLightBox[3], self.tafficLightBox[0]:self.tafficLightBox[2]]
        #     state = color_detect.detcet_color(img0)
        #     if (state != 0):
        #         self.tafficLightState = color_detect.detcet_color(img0)
        #     cv2.rectangle(img, (self.tafficLightBox[0], self.tafficLightBox[1]),
        #                   (self.tafficLightBox[2], self.tafficLightBox[3]), (0, 0, 255), 2)

        # print(self.tafficLightState)
        # tmp_car_number = 0


        for line in self.lines:
            line.get_velocity()
            #plot_one_box((line.box[0], line.box[1]), (line.box[2], line.box[3]), img, label=line.Classname, color=(255,0,0), line_thickness=3)

            if line.Classname == 'car' and line.velocity != 0:
                # show_velocity = "velocity:" + str(line.velocity) + 'm/s'
                # 速度显示
                show_velocity = str(line.velocity) + 'km/h'
                # cv2.putText(img, show_velocity, (int(line.line_points[-1][0]) + 50, int(line.line_points[-1][1])),
                #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

            print("line.Classname", line.Classname)
            if line.Classname == 'person': #car
                # tmp_car_number += 1
                last_points = 0
                for points in line.line_points:
                    # if(type(points) == int):
                    #     continue
                    # print("points", points)
                    # print("lastpoints", last_points)
                    if (type(points) == list and last_points != 0):
                        cv2.line(img, (int(last_points[0]), int(last_points[1])),
                                 (int(points[0]), int(points[1])), (245, 23, 167), 2)
                    last_points = points
            # car_number.append(tmp_car_number)
                    # cv.circle(self.img, (int(points[0]), int(points[1])), 1, (0, 0, 255), 4)
        return img

    def car_count(self,img):
        count = 0
        # print("111111")
        print("line_in", Recice_Line)

        for judge_line in Recice_Line:  # [[[].[]],[[].[]]]
            for line in self.lines:
                # print("line.isCount:", line.isCount)
                if (line.isCount == False):
                    # print("1:",judge_line[0][0], judge_line[0][1])
                    # print("2:",judge_line[1][0], judge_line[1][1])
                    # print("3:",line.line_points[0])
                    # print("4:",line.line_points[-1])
                    res = judge((judge_line[0][0], judge_line[0][1]),
                                (judge_line[1][0], judge_line[1][1]),
                                (int(line.line_points[0][0]), int(line.line_points[0][1])),
                                (int(line.line_points[-1][0]), int(line.line_points[-1][1])))

                    if (res == True):
                        # print("res:", res)
                        line.isCount = True
                        Return_Car[count] += 1

                        print("car:",Return_Car,count,res)
                        # cv2.putText(img, Return_Car[count],
                        #             (int(judge_line[0][0] + 5) + 50, int(judge_line[0][1] + 25)),
                        #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 4)
            count += 1


    def y_line(self, img):  # 实线压线违规
        count = 0
        # print("111111")
        print("line_in", Recice_Line)
        for judge_line in Recice_Line:  # [[[].[]],[[].[]]]
            img_count = 0
            for line in self.lines:
                res = judge((judge_line[0][0], judge_line[0][1]),
                            (judge_line[1][0], judge_line[1][1]),
                            (int(line.line_points[0][0]), int(line.line_points[0][1])),
                            (int(line.line_points[-1][0]), int(line.line_points[-1][1])))

                if (res == True):
                    #车违规
                    cv2.putText(img, " ",
                                (int(line.line_points[-1][0] + 5) + 50, int(line.line_points[-1][1] + 25)),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 4)
                    # 返回违规图像'%Y-%m-%d-%H-%M-%S'
                    img_name = "E:/Desktop/img/a+" + time.strftime('%Y-%m-%d-%H-%M-%S') + str(img_count) + "+.jpg"
                    # cv2.imwrite(img_name, img)
                    screenShot = True
                    # print("res:",res)
                    #line.isCount = True
                    img_count += 1
                    # Return_Car[count] += 1
            count += 1

    def bmx_d(self, img):  # 实线压线违规
        for line in self.lines:
            if (line.Classname == 'person'):
                res = bmx.quadrangle(Recive_Rect, [line.line_points[-1][0], line.line_points[-1][1]])
                print("res:", res)
                if (res == False):
                    #行人闯红灯
                    cv2.putText(img, " ",
                                (int(line.line_points[-1][0] + 5) + 50, int(line.line_points[-1][1] + 5)),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 4)

    def chuanghongdeng(self, img):
        self.tafficLightState = 3
        if (self.tafficLightState == 3):
            # print("!!!!!!!!!!!!!!!!!!!")
            for line in self.lines:
                if (line.Classname == 'motor'):
                    # print("??????????")
                    line.get_velocity_x()
                    # print("line.velocity", line.velocity_x)
                    if (line.velocity_x > 50):  ############调整阈值，并返回截图
                        # cv2.rectangle(img, (int(line.line_points[0][0]), int(line.line_points[0][1])), (int(line.line_points[0][0]) + 100, int(line.line_points[0][1]) + 100), (0,0,255), 2)
                        #摩托违规
                        cv2.putText(img, " ",
                                    (int(line.line_points[-1][0]) + 50, int(line.line_points[-1][1])),
                                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

    def car_number(self):
        count = 0
        for line in self.lines:
            if(line.Classname == "car"):
                count += 1
        return count
    # def resnet_calssify(self, image):
    #     for line in self.lines:
    #         if(line.Classname == 'car' and line.isClassify == False):
    #             image = Image.fromarray(image)
    #             resnet_img = image.crop(line.box)
    #             car_class = resnet_car(resnet_img)
    #             image = np.asarray(image)
    #             cv2.putText(image, car_class, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    #             pass
    #
    #     return image


class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        #QSS文件
        with open("./美化.qss") as f:
            self.setStyleSheet(f.read())
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./img/秋天背景.jpg")))
        self.setPalette(palette)

        # Hide Window Title
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        # VideoDisplay
        self.pushButton_1.clicked.connect(self.convert_status)
        self.open_flag = False
        self.video_stream = cv2.VideoCapture(0)
        self.painter = QPainter(self)

        # 定义编解码器并创建VideoWriter对象,保存视频

        self.filepath = './行车记录/recordVideo.avi'
        self.FPS = 24
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.filepath, self.fourcc,fps = self.FPS, frameSize = (1680,930))


        self.device = select_device('0')
        self.model = attempt_load('weights/yolov5x.pt', map_location=self.device)

        self.car = Car()

    def convert_status(self):
        if self.open_flag:
            print(self.open_flag)
        else:
            print(self.open_flag)
        self.open_flag = bool(1 - self.open_flag)
        ######
    def paintEvent(self, a0: QtGui.QPaintEvent):
        if not self.open_flag:
            print("Can't receive frame (stream end?)")
        if self.open_flag:
            ret, self.image = self.video_stream.read()


            self.out_img, self.car.box_points, self.car.all_points = PoseDect(self.image, self.model, imgsz=640)

            self.car.update_point()
            self.car.delete_point()
            self.out_img = self.car.draw_line(self.out_img)


            self.out_img = cv2.resize(self.out_img, (1500, 500), interpolation=cv2.INTER_AREA)#(1680, 930)
            self.out_img = cv2.cvtColor(self.out_img, cv2.COLOR_BGR2RGB)
           # frame = cv2.flip(frame, 1)  #视频水平翻转
            ret = self.out.write(self.out_img)
            self.Qframe = QImage(self.out_img.data, self.out_img.shape[1], self.out_img.shape[0], self.out_img.shape[1] * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))    #与label关联
            self.update()

            self.car.time_count += 1

    def messageDialog(self):
        #欢迎语，我觉得可以不加
        # self.msg_box = QMessageBox(QMessageBox.Information, '欢迎', '欢迎使用EScorter！')
        # self.msg_box.exec_()
        pass
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 动画部分
    splash = SplashScreen()
    splash.effect()
    app.processEvents()  #设置启动画面不影响其他效果cd ~/workspace/deepcar/deeplearning_python/src
    myWindow = MyWindow()
    myWindow.messageDialog() #消息提示框
    myWindow.show()
    splash.finish(myWindow)  # 启动画面完成启动
    sys.exit(app.exec_())
    cap.release()
    out.release()
    cv2.destroyAllWindows()

