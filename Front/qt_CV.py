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




class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture(0)  # 初始化摄像头
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

        self.device = select_device('0')
        self.model = attempt_load('weights/yolov5x.pt', map_location=self.device)

        self.car = Car()

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # QVBoxLayout类垂直地摆放小部件

        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_close = QtWidgets.QPushButton(u'退出')

        # button颜色修改
        button_color = [self.button_open_camera, self.button_close]
        for i in range(2):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{border_radius:10px}"
                                           "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        # move()方法是移动窗口在屏幕上的位置到x = 500，y = 500的位置上
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'摄像头')

        '''
        # 设置背景颜色
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(),QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''

    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #                     pass
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开相机')

    def show_camera(self):

        flag, self.image = self.cap.read()

        print(self.image)
        print(type(self.image))
        #self.image = cv2.imread(r"D:\SO\yolov5\data\images\1.png")
        #self.image是读取到的图
        #将读取到的图片接入yolov5进行推理
        # np.set_printoptions(threshold=np.inf)
        # print('==========================================================================================')
        # print(self.image.dtype)

        # transf = transforms.ToTensor()
        # self.img_tensor = transf(self.image)

        # self.car.box_points = []
        # self.car.all_points = []
        #cn = self.car.car_number()

        #self.out_img = PoseDect(self.image, self.model, imgsz=640)
        self.out_img, self.car.box_points, self.car.all_points = PoseDect(self.image,self.model, imgsz=640)

        self.car.update_point()
        self.car.delete_point()
        self.out_img = self.car.draw_line(self.out_img)
        print("lines", self.car.lines)

        # self.car.chuanghongdeng(self.out_img)
        # self.car.car_count(self.out_img)
        # self.car.y_line(self.out_img)
        # self.car.bmx_d(self.out_img)

        # self.last_img  推理后的结果
        # show = cv2.resize(self.out_img, (640, 640))
        # self.image.dtype = 'uint8'
        show = cv2.resize(self.out_img, (640, 640))

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

        self.car.time_count += 1###

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()



if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())


