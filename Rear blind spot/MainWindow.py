# -*- coding: utf-8 -*-
# @Time    : 2021/4/19 22:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : MainWindow.py
# @Software: PyCharm。
import sys,cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import time
#窗口控件
from ui_MainWindow3 import Ui_MainWindow
#初始动画
from initial_animation import SplashScreen
#定时框架
# from apscheduler.schedulers.blocking import BlockingScheduler
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar , load_pretrained_model
from paddleseg.cvlibs import manager, Config

from predict_two import predict

from paddleseg.cvlibs import  Config

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
        self.record = False
        self.pushButton_2.clicked.connect(self.convert_capture)

        self.model_path =  './models/model2.pdparams'

        self.cfg = Config('./configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml')

        self.model = self.cfg.model

        # self.a = json.loads(self.model)
        # self.b = json.loads(self.model_path)
        #
        # self.i = utils.utils.load_entire_model(self.a , self.b)

        load_pretrained_model(self.model, self.model_path)
        #追踪model来源

        utils.utils.load_entire_model(self.model, self.model_path)

        self.model.eval()

    def convert_status(self):
        # if self.open_flag:
        #     print(self.open_flag)
        # else:
        #     print(self.open_flag)
        self.open_flag = bool(1 - self.open_flag)

    def convert_capture(self):
        # if self.record:
        #     print('record',1-self.record)
        #     self.lineEdit.setText("温馨提示：行车记录已开启")
        # else:
        #     print('record',1-self.record)
        #     self.lineEdit.setText("温馨提示：行车记录已关闭")
        self.record = bool(1 - self.record)

    def paintEvent(self, a0: QtGui.QPaintEvent):
        if not self.open_flag:
            # print("Can't receive frame (stream end?)")
            self.lineEdit.setText("温馨提示：摄像头已关闭")
        if self.open_flag:
            self.lineEdit.setText("温馨提示：摄像头已开启")

            ret, self.image = self.video_stream.read()

            self.out_img = predict(self.model, self.image, self.cfg)

            size = self.out_img.shape
            # print(size)  # (1728, 3072, 3)

            ######################################################################################################
            # 画线

            cv2.line(self.out_img, pt1=(181, 290), pt2=(449, 290), color=(0, 255, 255), thickness=2) #中间

            cv2.line(self.out_img, pt1=(385, 190), pt2=(240, 190), color=(0, 255, 0), thickness=2) #最外侧
            #
            cv2.line(self.out_img, pt1=(125, 390), pt2=(520, 390), color=(0, 0, 255), thickness=2) #最内侧
            #
            cv2.line(self.out_img, pt1=(580, 480), pt2=(385, 190), color=(255, 191, 0), thickness=2)
            #
            cv2.line(self.out_img, pt1=(60, 480), pt2=(240, 190), color=(255, 191, 0), thickness=2)


            frame = cv2.resize(self.out_img, (1680, 930), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(self.out_img, cv2.COLOR_BGR2RGB)
           # frame = cv2.flip(frame, 1)  #视频水平翻转

            if self.record:
                # print("Capture Stop!")
                self.lineEdit.setText("温馨提示：行车记录已开启")
            if not self.record:
                self.lineEdit.setText("温馨提示：行车记录已关闭")
                ret = self.out.write(frame)

            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))    #与label关联
            self.update()

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
    app.processEvents()  #设置启动画面不影响其他效果
    myWindow = MyWindow()
    myWindow.messageDialog() #消息提示框
    myWindow.show()
    splash.finish(myWindow)  # 启动画面完成启动
    sys.exit(app.exec_())
    cap.release()
    out.release()
    cv2.destroyAllWindows()

