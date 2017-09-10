#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QEvent>
#include <QMouseEvent>
#include <QPoint>
#include <QFileDialog>
#include <QDebug>
#include <QPainter>
#include <QGraphicsScene>
#include <QMessageBox>
#include <QTime>
#include <opencv2/core/core.hpp>
#include <QElapsedTimer>
#include "classification.h"
#include "multi_class.h"

using namespace cv_class;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    bool    eventFilter(QObject *obj, QEvent *event);      // event filter
    void    clear();
    void    draw_line(float A, float B, float C);
    int     rand_num(int max);
    void    show_result(float A, float B, float C, float thres=0);
    void    show_result(float thres, QString filename);
    bool    is_multiclass_enabled();
    void    clear_line();
    void    draw_skywater_line(cv::Mat &img);

private slots:
    void on_img1_clicked();
    void on_img2_clicked();
    void on_img3_clicked();
    void on_process_clicked();
    void on_radioButton_rg_clicked();
    void on_radioButton_rb_clicked();
    void on_next_clicked();
    void on_comboBox_activated(int index);

private:
    Ui::MainWindow          *ui;
    cv::Mat                 cv_img_;
    QPoint                  cord_;
    QImage                  q_img_;
    QGraphicsScene          *scene_;
    vector<QPointF>         man_pt_;
    int                     mode_;
    cv_classification       single_class_;
    multi_class             multi_class_;
    QPen                    color_pen_;
    QElapsedTimer           timer_;
    qint64                  nanoSec_;
    vector<cv::Vec2i>       lq_pts_;        // least square test pts
    cv::Vec4f               lq_result_;     // least square result: (w0,w1,x0,x1)

protected:
//     void paintEvent(QPaintEvent *event);
};

#endif // MAINWINDOW_H
