#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QGraphicsPixmapItem>

/* DO NOT EDIT */
// color index
#define     RED         0
#define     GREEN       1
#define     BLUE        2
// combo box number
#define     c_RGB_THR     0
#define     c_LDA         1
#define     c_SVM         2
#define     c_MANUAL      3
#define     c_MY_CAL      4
/* DO NOT EDIT */

using namespace std;
using namespace cv;

unsigned int img_flag = 0;
unsigned int color_num = 0;
unsigned int index1,index2;

MainWindow::MainWindow(QWidget *parent) :QMainWindow(parent), ui(new Ui::MainWindow)
{
    mode_ = 0;
    nanoSec_ = 0;
    man_pt_.reserve(10);
    single_class_.setIndex1(RED);
    single_class_.setIndex2(GREEN);
    multi_class_.setIndex1(RED);
    multi_class_.setIndex2(GREEN);
    scene_ = new QGraphicsScene(ui->graphicsView);
    color_pen_.setColor(QColor(color_num ,color_num ,color_num ));
    color_pen_.setWidth(3);

    ui->setupUi(this);
    ui->initial_pic->installEventFilter(this);
    ui->graphicsView->installEventFilter(this);
    ui->graphicsView->setScene(scene_);
    ui->comboBox->addItem("1 RGB threshold");
    ui->comboBox->addItem("2 Linear Disciminant Analysis(LDA)");
    ui->comboBox->addItem("3 Support Vector Machine(SVM)");
    ui->comboBox->addItem("4 Manual");
    ui->comboBox->addItem("5 my line calculation");

    on_img1_clicked();      // show the first image
    clear();
    //process_class_.test();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::clear()
{
    // Graphics view initialization
    scene_->clear();
    scene_->setSceneRect(0,0,255,255);
    scene_->addEllipse(0,0,2,2);
    QPen red_pen(QColor(255,0,0));
    QPen green_pen(QColor(0,255,0));
    QPen blue_pen(QColor(0,0,255));
    red_pen.setWidth(3);
    green_pen.setWidth(3);
    blue_pen.setWidth(3);

    scene_->addLine(0,0,255,0, red_pen);
    if(single_class_.index2() == GREEN)
        scene_->addLine(0,0,0,255, green_pen);
    else
        scene_->addLine(0,0,0,255, blue_pen);
    ui->graphicsView->setScene(scene_);
    //ui->graphicsView->show();

    // clear image
    ui->process_pic->clear();

    // clear data
    single_class_.clear();
    multi_class_.clear();
    man_pt_.clear();
}

void MainWindow::clear_line()
{
    // Graphics view initialization
    scene_->clear();
    scene_->setSceneRect(0,0,255,255);
    scene_->addEllipse(0,0,2,2);
    QPen red_pen(QColor(255,0,0));
    QPen green_pen(QColor(0,255,0));
    QPen blue_pen(QColor(0,0,255));
    red_pen.setWidth(3);
    green_pen.setWidth(3);
    blue_pen.setWidth(3);

    scene_->addLine(0,0,255,0, red_pen);
    if(single_class_.index2() == GREEN)
        scene_->addLine(0,0,0,255, green_pen);
    else
        scene_->addLine(0,0,0,255, blue_pen);
    ui->graphicsView->setScene(scene_);

    QPen good_p(QColor(255,0,255)); good_p.setWidth(3);
    QPen bad_p(QColor(0,0,0)); bad_p.setWidth(3);
    vector<cv::Vec3f> tmp_vec = single_class_.get_good_vec();
    vector<cv::Vec3f> tmp_vec2 = single_class_.get_bad_vec();
    vector<cv::Vec2f> tmp_vec3 = multi_class_.get_pts_vec();

    // draw points
    if(tmp_vec.size() != 0 && tmp_vec2.size() != 0)
    {
        for(cv::Vec3f pt : tmp_vec)
        {
            scene_->addEllipse(pt.val[index1],pt.val[index2],2,2,good_p );
        }

        for(cv::Vec3f pt : tmp_vec2)
        {
            scene_->addEllipse(pt.val[index1],pt.val[index2],2,2,bad_p );
        }
    }
    else if(tmp_vec3.size() != 0)
    {
        for(cv::Vec2f pt : tmp_vec3)
        {
            scene_->addEllipse(pt[0],pt[1],2,2,bad_p );
        }
    }
    else
        qDebug()<<"Error: no points to draw!";
}

void MainWindow::draw_skywater_line(Mat &img)
{
    cv::Vec3b lgreen2(102,255,178),grey(160,160,160),black(0,0,0);
    for (int i = 0; i < img.cols; ++i)
    {
        for (int j = 0; j < img.rows-10; ++j)
        {
            cv::Vec3b rgb=img.at<cv::Vec3b>(j,i);
            cv::Vec3b rgb2=img.at<cv::Vec3b>(j+1,i);
            if(rgb == grey && rgb2 == lgreen2)
            {
                //img.at<cv::Vec3b>(j,i) = black;
                //img.at<cv::Vec3b>(j+1,i) = black;
                lq_pts_.push_back(cv::Vec2i(i,j));  // !! col is x-axis; row is y-axis
                break;
            }
        }
    }


    fitLine(lq_pts_,lq_result_,CV_DIST_L2, 0, 0.01, 0.01);
    qDebug("w:[%f %f] x0:[%f %f]",lq_result_[0],lq_result_[1],lq_result_[2],lq_result_[3]);

    cv::Point2d w( lq_result_[0],lq_result_[1] ), pt0(lq_result_[2],lq_result_[3] );
    cv::Point2d pt1 = pt0 + 1000*w, pt2 = pt0 - 1000*w;

    line(img,pt1,pt2,CV_RGB(0,0,0),2,8);

}

void MainWindow::draw_line(float A, float B, float C)
{
    cv::Vec2f   pt1(0,0), pt2(0,0);
    if(A != 0)
    {
        pt1[0] = -C/A;
        pt1[1] = 0;
        pt2[0] = (-C-B*255)/A;
        pt2[1] = 255;
    }
    else if(B != 0)
    {
        pt1[0] = 0;
        pt1[1] = -C/B;
        pt2[0] = 255;
        pt2[1] = -C/B;
    }
    else
        return;
    scene_->addLine(pt1[0],pt1[1], pt2[0], pt2[1]);
}

int MainWindow::rand_num(int max)
{
    QTime time;
    time= QTime::currentTime();
    qsrand(time.msec()+time.second()*1000);

    return qrand() % max;
}

void MainWindow::on_img1_clicked()
{
    img_flag = 1;

    cv_img_ = cv::imread("../img/1");
    // change from BGR to RGB
    cvtColor(cv_img_, cv_img_, CV_BGR2RGB);  //cv::flip(image_, image_, 1);
    q_img_ = QImage((const unsigned char*)(cv_img_.data), cv_img_.cols, cv_img_.rows,
                            QImage::Format_RGB888);
    ui->initial_pic->setPixmap(QPixmap::fromImage(q_img_));
    ui->initial_pic->resize(ui->initial_pic->pixmap()->size());
    ui->note->setText("NOTE: Left mouse button  and right mouse button to select 2 classes of points");
    ui->next->setDisabled(1);

    clear();
}

void MainWindow::on_img2_clicked()
{
    img_flag = 2;   
    cv_img_ = cv::imread("../img/2");
    // cv::resize(cv_img_,cv_img_,Size(332,234));
    // change from BGR to RGB
    cvtColor(cv_img_, cv_img_, CV_BGR2RGB);  //cv::flip(image_, image_, 1);
    q_img_ = QImage((const unsigned char*)(cv_img_.data), cv_img_.cols, cv_img_.rows,
                            QImage::Format_RGB888);
    ui->initial_pic->setPixmap(QPixmap::fromImage(q_img_));
    ui->initial_pic->resize(ui->initial_pic->pixmap()->size());
    //ui->initial_pic->resize(QSize(332,234));
    ui->note->setText("NOTE: Please press 'next class' to select different classes of points");

    if(is_multiclass_enabled())
        ui->next->setDisabled(0);
    else
        ui->next->setDisabled(1);

    clear();
}

void MainWindow::on_img3_clicked()
{
    img_flag = 3;
    cv_img_ = cv::imread("../img/5");
    float scale_fac = 0.8;
    cv::resize(cv_img_,cv_img_, Size(0,0), scale_fac, scale_fac);
    // change from BGR to RGB
    cvtColor(cv_img_, cv_img_, CV_BGR2RGB);  //cv::flip(image_, image_, 1);
    q_img_ = QImage((const unsigned char*)(cv_img_.data), cv_img_.cols, cv_img_.rows,
                            QImage::Format_RGB888);
    ui->initial_pic->setPixmap(QPixmap::fromImage(q_img_));
    ui->initial_pic->resize(ui->initial_pic->pixmap()->size());
    ui->note->setText("NOTE: Please press 'next class' to select different classes of points");

    if(is_multiclass_enabled())
        ui->next->setDisabled(0);
    else
        ui->next->setDisabled(1);

    clear();
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    mode_ = ui->comboBox->currentIndex();
    if(obj == ui->initial_pic)
    {
        if(event->type() == QEvent::MouseButtonPress)
        {
            // get index(RG or RB)
            QMouseEvent *MouseEvent = static_cast<QMouseEvent*>(event);
            //QGraphicsView *view = ui->graphicsView;
            cord_               =  MouseEvent ->pos();
            index1 = single_class_.index1();
            index2 = single_class_.index2();
            QColor              c = q_img_.pixel(cord_);
            int r=c.red(), g=c.green(), b=c.blue();
            Vec3f pt(r,g,b);

            // display coordinate and RGB
            ui->R->setText("R : "+ QString::number(r));
            ui->G->setText("G : "+ QString::number(g));
            ui->B->setText("B : "+ QString::number(b));
            ui->x->setText("x : "+ QString::number(cord_.rx()));
            ui->y->setText("y : "+ QString::number(cord_.ry()));

            // save points
            if(is_multiclass_enabled())
            {
                scene_->addEllipse(pt[index1],pt[index2],2,2,color_pen_ );
                //view->show();    // image size: 332*234

                cv::Vec2f ptt(pt[index1],pt[index2]);
                multi_class_.push_pts(ptt);
                return true;
            }
            else
            {
                if( MouseEvent->button() == Qt::LeftButton)
                {
                    QPen p(QColor(255,0,255)); p.setWidth(3);
                    scene_->addEllipse(pt.val[index1],pt.val[index2],2,2,p );
                    //view->show();    // image size: 332*234

                    single_class_.push_good_pt(pt);
                    return true;
                }
                else if( MouseEvent->button() == Qt::RightButton)
                {
                    QPen p(QColor(0,0,0)); p.setWidth(3);
                    scene_->addEllipse(pt.val[index1],pt.val[index2],2,2,p );
                    //view->show();

                    single_class_.push_bad_pt(pt);
                    return true;
                }
            }
        }
        else return false;
    }
    else if(obj == ui->graphicsView)
    {
        if(event->type() == QEvent::MouseButtonPress)
        {
            QGraphicsView *view = ui->graphicsView;
            QMouseEvent *MouseEvent = static_cast<QMouseEvent*>(event);

            // display coordinate
            cord_ =  MouseEvent ->pos();
            QPointF scene_pt = view->mapToScene(cord_);             // !! transform coordinates from view to scene
            float x = scene_pt.rx(), y = scene_pt.ry();
            ui->x->setText("x : "+ QString::number(x));
            ui->y->setText("y : "+ QString::number(y));

            if(is_multiclass_enabled())
            {
                cv::Vec2f   ptt(x,y);
                multi_class_.push_manual(ptt);
                scene_->addEllipse(x,y,1,1,color_pen_ );

                unsigned int size=multi_class_.manual_vec_.size();
                if(size%2 == 0)
                {
                    cv::Vec2f pt1 = multi_class_.manual_vec_.at(size-1);
                    cv::Vec2f pt2 = multi_class_.manual_vec_.at(size-2);
                    float A,B,C;
                    single_class_.cal_line(pt1,pt2,A,B,C);
                    draw_line(A,B,C);
                }
                //view->show();

                return true;
            }
            else
            {
                QPen p(QColor(123,0,255)); p.setWidth(3);

                if(mode_ == c_MANUAL)      // manual mode
                {
                    man_pt_.push_back(scene_pt);
                    unsigned int size = man_pt_.size();

                    if(size<=2)
                    {
                        scene_->addEllipse(x,y,3,3,p );
                        if(size == 2)
                        {
                            float A,B,C;
                            cv::Vec2f pt1(man_pt_.at(0).rx(),man_pt_.at(0).ry());
                            cv::Vec2f pt2(man_pt_.at(1).rx(),man_pt_.at(1).ry());
                            single_class_.cal_line(pt1,pt2,A,B,C);
                            draw_line(A,B,C);
                        }
                    }
                    //view->show();

                }
                return true;
            }
        }
    }
    else return MainWindow::eventFilter(obj, event);
}

void MainWindow::on_process_clicked()
{
    mode_ = ui->comboBox->currentIndex();
    qDebug("Current selection:%d",mode_);

    float thres = 0.0, k=0, b=0, A=0, B=0, C=0, x0=0, y0=0;
    Eigen::Vector2f w(0,0);
    cv::Mat img;
    QImage q_img;
    cv_img_.copyTo(img);

    switch(mode_)
    {
        case c_RGB_THR:
            clear_line();
            timer_.start();

            thres = single_class_.cal_thres(0.42);

            nanoSec_ = timer_.nsecsElapsed();
            qDebug()<<"color_thres training time(ns):"<<nanoSec_;

            show_result(thres, QString::number(0.42)+".bmp");
            break;

        case c_MY_CAL:
            clear_line();
            timer_.start();

            single_class_.my_own_classify(k,b,x0,y0);

            nanoSec_ = timer_.nsecsElapsed();
            qDebug()<<"my_cal training time(ns):"<<nanoSec_;

            A=-k; B=1; C=-b;
            show_result(-A,-B,-C);
            scene_->addEllipse(x0,y0,5,5 );
            break;

        case c_LDA:
            clear_line();
            single_class_.LDA(w, thres);
            A=w[0]; B=w[1]; C=0;
            show_result(A,B,C,thres);
            break;

        case c_SVM:
            if(1==img_flag)
            {
                clear_line();
                single_class_.SVM(img);

                cvtColor(img,img, CV_BGR2RGB);
                q_img = QImage((const unsigned char*)(img.data), img.cols, img.rows,
                                        QImage::Format_RGB888);
                ui->process_pic->setPixmap(QPixmap::fromImage(q_img));
                ui->process_pic->resize(ui->process_pic->pixmap()->size());
            }
            else if(2==img_flag)
            {
                timer_.start();
                multi_class_.SVM(img);
                nanoSec_ = timer_.nsecsElapsed();
                qDebug()<<"time elapsed:"<<nanoSec_<<" ns";

                cvtColor(img,img, CV_BGR2RGB);
                q_img = QImage((const unsigned char*)(img.data), img.cols, img.rows,
                                        QImage::Format_RGB888);
                ui->process_pic->setPixmap(QPixmap::fromImage(q_img));
                ui->process_pic->resize(ui->process_pic->pixmap()->size());
            }
            else if(3==img_flag)
            {
                timer_.start();
                multi_class_.SVM(img);
                nanoSec_ = timer_.nsecsElapsed();
                qDebug()<<"time elapsed:"<<nanoSec_<<" ns";

                draw_skywater_line(img);
                cv::resize(img,img,Size(405,324));
                cv::imwrite("skywater.png",img);

                cvtColor(img,img, CV_BGR2RGB);
                q_img = QImage((const unsigned char*)(img.data), img.cols, img.rows,
                                        QImage::Format_RGB888);
                ui->process_pic->setPixmap(QPixmap::fromImage(q_img));
                ui->process_pic->resize(ui->process_pic->pixmap()->size());
            }
            break;

        case c_MANUAL:
            if(1==img_flag)
            {
                clear_line();
                if(man_pt_.size()>=2)
                {
                    cv::Vec2f pt1(man_pt_.at(0).rx(),man_pt_.at(0).ry());
                    cv::Vec2f pt2(man_pt_.at(1).rx(),man_pt_.at(1).ry());

                    timer_.start();
                    single_class_.cal_line(pt1,pt2,A,B,C);

                    nanoSec_ = timer_.nsecsElapsed();
                    qDebug()<<"manual training time(ns):"<<nanoSec_;

                    show_result(-A,-B,-C);
                }
                else
                {
                    QMessageBox::information(this, "Look here!",
                    QString("Please manually select two points in the coordinate frame to form a line!"));
                    return;
                }
            }
            else
            {
                QImage process_img(q_img_);
                /// training
                timer_.start();

                multi_class_.train_manual();

                nanoSec_ = timer_.nsecsElapsed();
                qDebug()<<"manual training time(ns):"<<nanoSec_;

                /// testing
                timer_.start();
                for(int i=0; i<process_img.width();i++)
                {
                    for(int j=0; j<process_img.height();j++)
                    {
                        QColor c = process_img.pixel(i,j);
                        float x=c.red();
                        float y= (index2 == GREEN)? c.green():c.blue();
                        cv::Vec2f pt(x,y);

                        unsigned int type = multi_class_.classify(pt);
                        if(type == 1)
                            process_img.setPixel(i,j,qRgb(255,102,102));    // dark magenta
                        else if(type == 2)
                            process_img.setPixel(i,j,qRgb(160,160,160));    // grey
                        else if(type==3)
                            process_img.setPixel(i,j,qRgb(178,255,102));    // dark green
                        else
                            process_img.setPixel(i,j,qRgb(255,255,255));    // white
                    }
                }
                nanoSec_ = timer_.nsecsElapsed();
                qDebug()<<"manual testing time(ns):"<<nanoSec_;

                process_img.save("result.bmp");
                qDebug()<<"error rate:"<<multi_class_.error_rate()*100<<"%";
                ui->process_pic->setPixmap(QPixmap::fromImage(process_img));
                ui->process_pic->resize(ui->initial_pic->pixmap()->size());
            }
            break;
        default:
            break;
    }
}

void MainWindow::on_radioButton_rg_clicked()
{
    if(ui->radioButton_rg->isChecked())
    {
        index1=RED; index2=GREEN;
        single_class_.setIndex1(RED);
        single_class_.setIndex2(GREEN);
        multi_class_.setIndex1(RED);
        multi_class_.setIndex2(GREEN);
        clear();
        qDebug()<<"set to red and green";
    }
}

void MainWindow::on_radioButton_rb_clicked()
{
    if(ui->radioButton_rb->isChecked())
    {
        index1=RED; index2=BLUE;
        single_class_.setIndex1(RED);
        single_class_.setIndex2(BLUE);
        multi_class_.setIndex1(RED);
        multi_class_.setIndex2(BLUE);
        clear();
        qDebug()<<"set to red and blue";
    }
}

void MainWindow::on_next_clicked()
{
    color_num = color_num+50 > 255? (color_num/11) : (color_num + 50);
    int r=0,g=0,b=0;
    r=rand_num(255); g=rand_num(500)/2; b=rand_num(1000)/4;
    color_pen_.setColor(QColor(r,g,b));
    qDebug("r%d g%d b%d",r,g,b);
    qDebug()<<"class:"<<multi_class_.next_label();
}

void MainWindow::show_result(float A, float B, float C, float thres)
{
    qDebug("line equation:%fx+%fy+%f=0",A,B,C);
    draw_line(A,B,C);

    timer_.start();

    QImage process_img(q_img_);
    for(int i=0; i<process_img.width();i++)
    {
        for(int j=0; j<process_img.height();j++)
        {
            QColor c = process_img.pixel(i,j);
            float x=c.red();
            float y= index2 == GREEN? c.green():c.blue();

            if(A*x+B*y+C > thres)
                process_img.setPixel(i,j,qRgb(255,0,0));
            else
                process_img.setPixel(i,j,qRgb(255,255,255));
        }
    }
    nanoSec_ = timer_.nsecsElapsed();
    qDebug()<<"testing uses time(ns):"<<nanoSec_;

    process_img.save("result.bmp");
    ui->process_pic->setPixmap(QPixmap::fromImage(process_img));
    ui->process_pic->resize(ui->initial_pic->pixmap()->size());
}

void MainWindow::show_result(float thres,QString filename)
{

    timer_.start();

    QImage process_img(q_img_);
    for(int i=0; i<process_img.width();i++)
    {
        for(int j=0; j<process_img.height();j++)
        {
            QColor c = process_img.pixel(i,j);
            float r=c.redF(), g=c.greenF(), b=c.blueF();

            if(r/(r+g+b) > thres)
                process_img.setPixel(i,j,qRgb(255,0,0));
            else
                process_img.setPixel(i,j,qRgb(255,255,255));
        }
    }
    nanoSec_ = timer_.nsecsElapsed();
    qDebug()<<"testing uses time(ns):"<<nanoSec_;

    process_img.save("result.bmp");
    ui->process_pic->setPixmap(QPixmap::fromImage(process_img));
    ui->process_pic->resize(ui->initial_pic->pixmap()->size());
}

bool MainWindow::is_multiclass_enabled()
{
    mode_ = ui->comboBox->currentIndex();
    if((img_flag == 2 || img_flag == 3) && (mode_ == c_SVM || mode_ == c_MANUAL))
        return true;
    else
        return false;
}

void MainWindow::on_comboBox_activated(int index)
{
    if((2==img_flag || img_flag == 3) && (c_SVM==index || c_MANUAL==index))
        ui->next->setDisabled(0);
    else
        ui->next->setDisabled(1);
}
