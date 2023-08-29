# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_MainWindow3.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap,QPalette, QBrush
from PyQt5.QtCore import pyqtSignal,pyqtSlot,Qt,QThread
from PyQt5.QtWidgets import QMainWindow,QApplication,QDesktopWidget
import cv2
import numpy as np
import time
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1130)
        # Hide Window Title
        MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1680, 960))
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1130))

        MainWindow.setWindowIcon(QIcon('./img/windowicon.png'))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1105))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setStyleSheet("font: 24pt \"新宋体\";\n"
"")
        self.label.setObjectName("label")
        self.label.setPixmap(QtGui.QPixmap("./img/4D背景.jpg"))
        # self.label.setPixmap(QtGui.QPixmap("./img/label.jpg"))
        #self.label.setPixmap(QtGui.QPixmap("./img/道路1920x1278.png"))
        self.label.setScaledContents(True)

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)

        self.lineEdit.setReadOnly(True) #设置为只读
        # self.lineEdit.setGeometry(QtCore.QRect(120, 70, 1680, 90))
        self.lineEdit.setGeometry(QtCore.QRect(180, 90, 840, 90))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.lineEdit.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(24)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        ###
        self.layoutWidget.setGeometry(QtCore.QRect(242, 870, 1436, 70))
        ###
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        ###
        self.horizontalLayout_2.setSpacing(350)
        ###
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_1 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_1.setMinimumSize(QtCore.QSize(95, 60))
        self.pushButton_1.setMaximumSize(QtCore.QSize(250, 250))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.horizontalLayout_2.addWidget(self.pushButton_1)
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(95, 0))
        self.pushButton_2.setMaximumSize(QtCore.QSize(250, 250))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(95, 60))
        self.pushButton_3.setMaximumSize(QtCore.QSize(250, 250))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_4.setMinimumSize(QtCore.QSize(95, 60))
        self.pushButton_4.setMaximumSize(QtCore.QSize(250, 250))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.label_2.setGeometry(QtCore.QRect(50, 20, 91, 31))
        self.label_2.setGeometry(QtCore.QRect(65, 55, 140, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(30) #20
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1680, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 时钟-----------------------------------
        self.lcd = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcd.setGeometry(QtCore.QRect(1370,55, 340, 90))
        self.lcd.setDigitCount(10)
        self.lcd.setMode(QLCDNumber.Dec)
        self.lcd.setSegmentStyle(QLCDNumber.Filled)
        self.lcd.setStyleSheet("border:0px") #消除边框
        self.lcd.display(time.strftime("%X", time.localtime()))


        # 新建一个QTimer对象
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.start()

        # 信号连接到槽
        self.timer.timeout.connect(self.onTimerOut)
        # -----------------------------------------------

        self.retranslateUi(MainWindow)
        self.pushButton_4.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # 定义时钟槽
    def onTimerOut(self):
        self.lcd.display(time.strftime("%X", time.localtime()))

    # 退出弹窗
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', '确认退出?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "")) #空值，否则无法正常显示pixmap！
       # self.lineEdit.setText(_translate("MainWindow", "  温馨提示：")) #提示框默认值
        self.lineEdit.setText(_translate("MainWindow", ""))
        self.pushButton_1.setText(_translate("MainWindow", ""))
        self.pushButton_2.setText(_translate("MainWindow", ""))
        self.pushButton_3.setText(_translate("MainWindow", ""))
        self.pushButton_4.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Escorter"))

if __name__=="__main__":
    app = QApplication(sys.argv)
    mywindow = QMainWindow()

    a = Ui_MainWindow()
    a.setupUi(mywindow)
    with open("./美化.qss") as f:
        mywindow.setStyleSheet(f.read())
    palette = QPalette()
    palette.setBrush(QPalette.Background, QBrush(QPixmap("./img/秋天背景.jpg")))
    mywindow.setPalette(palette)
    mywindow.show()
    sys.exit(app.exec_())