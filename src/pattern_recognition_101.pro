#-------------------------------------------------
#
# Project created by QtCreator 2016-03-07T16:16:21
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = pattern_recognition_101
TEMPLATE = app
CONFIG += c++11


SOURCES += main.cpp\
        mainwindow.cpp \
    classification.cpp \
    multi_class.cpp

HEADERS  += mainwindow.h \
    classification.h \
    multi_class.h

FORMS    += mainwindow.ui

INCLUDEPATH +=  /usr/include/opencv2    \
                /usr/incllude/eigen3

LIBS        +=  -L/usr/lib/x86_64-linux-gnu \
                -lopencv_core   \
                -lopencv_highgui    \
                -lopencv_imgproc    \
                -lopencv_ml
