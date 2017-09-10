#include "multi_class.h"
#include <iostream>
#include <QElapsedTimer>
#include <QDebug>

using namespace std;

qint64          m_nanoSec=0;
QElapsedTimer   m_timer;

multi_class::multi_class():label_(1), preprocess_(0)
{
    pts_vec_.reserve(100);
    label_vec_.reserve(100);
    mid_vec_.reserve(100);
    line_vec_.reserve(100);

    pts_vec_.clear();
    label_vec_.clear();
    mid_vec_.clear();
    line_vec_.clear();

}

multi_class::~multi_class()
{}

void multi_class::push_pts(cv::Vec2f &pt)
{
    pts_vec_.push_back(pt);
    label_vec_.push_back(label_);
}

void multi_class::push_manual(cv::Vec2f &pt)
{
    manual_vec_.push_back(pt);
}

unsigned int multi_class::next_label()
{
    return ++label_;
}

unsigned int multi_class::get_label()
{
    return label_;
}

int multi_class::rand_num(int max)
{
    QTime time;
    time= QTime::currentTime();
    qsrand(time.msec()+time.second()*1000);

    return qrand() % max;
}

void multi_class::clear()
{
    pts_vec_.clear();
    label_vec_.clear();
    mid_vec_.clear();
    line_vec_.clear();
    label_ = 1;
    preprocess_ = 0;
}

void multi_class::cal_mid_pt()
{
    unsigned int label = 1;
    int count = 0;
    cv::Vec2f ave(0,0);

    for(unsigned int i=0;i<pts_vec_.size();i++)
    {
        if(label_vec_.at(i) == label)
        {
            ave += pts_vec_.at(i);
            count++;
        }
        else
        {
            ave = ave/count;
            mid_vec_.push_back(ave);

            ave = pts_vec_.at(i);
            label++;
            count = 1;
        }
    }
    // last one mid point
    ave = ave/count;
    mid_vec_.push_back(ave);
}

void multi_class::cal_manual_line()
{
    float A=0,B=0,C=0;
    int flag=0;
    cv::Vec3f line_param(0,0,0);

    for(unsigned int i=0;i<manual_vec_.size()-1;i=i+2)
    {
        // Calculate A,B,C
        cal_line(manual_vec_.at(i),manual_vec_.at(i+1),A,B,C);

        if(mid_vec_.size()>=3)  // it means >=3 classes
        {
            for(int j=0;j<3;j++)
            {
                int line_flag = 0;
                float x = mid_vec_.at(j)[0], y = mid_vec_.at(j)[1];

                line_flag = (A*x+B*y+C>=0) ? 1 : -1;
                flag = flag + line_flag;
            }

            // possible flag value: 3, -3, 1, -1
            if(flag == 3)       // it means if A*x+B*y+C<0, then determines one class; so i need to change signs of A,B,C
            {
                line_param=cv::Vec3f(-A,-B,-C);
            }
            else if(flag == -3)
            {
                line_param=cv::Vec3f(A,B,C);
            }
            else
            {
                line_param=cv::Vec3f(-flag*A,-flag*B,-flag*C);
            }
            line_vec_.push_back(line_param);
        }
        else        // two classes classification; we don't care the sign of A,B,C
        {
            line_param=cv::Vec3f(A,B,C);
            line_vec_.push_back(-line_param);       // because scene frame is y-axis upside down
            line_vec_.push_back(line_param);
        }

        A=line_param[0]; B=line_param[1]; C=line_param[2];
        //cout<<"A:"<<A<<" B:"<<B<<" C:"<<C<<endl;
        flag = 0;
    }
}

void multi_class::train_manual()
{
    cal_mid_pt();
    cal_manual_line();
}

unsigned int multi_class::classify(cv::Vec2f &pt)
{
    for(unsigned int i=0;i<line_vec_.size();i++)
    {
        float A=line_vec_.at(i)[0], B=line_vec_.at(i)[1], C=line_vec_.at(i)[2];
        if(A*pt[0]+B*pt[1]+C>=0)
            return i+1;
    }

    return -1;

}

void multi_class::SVM(cv::Mat &img)
{
    cv::Mat svm_img = cv::Mat::zeros(256, 256, CV_8UC3);

    /// Set up training data
    int      size = pts_vec_.size();
    float    trainingData[size][2];
    float    labels[size];

    for(int i=0; i<size; i++)
    {
        labels[i] = label_vec_.at(i);
        trainingData[i][0] = pts_vec_.at(i)[0];
        trainingData[i][1] = pts_vec_.at(i)[1];
    }
    cv::Mat labelsMat(size, 1, CV_32FC1, labels);
    cv::Mat trainingDataMat(size, 2, CV_32FC1, trainingData);

    /// Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    params.C = 1;
    params.gamma = 0.001;

    /// Train the SVM
    m_timer.start();

    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

    m_nanoSec = m_timer.nsecsElapsed();
    qDebug()<<"SVM training time(ns):"<<m_nanoSec;

    /// Test image response
    m_timer.start();

    cv::Vec3b magenta(204,204,255), cyan(255,255,204), lgreen(204,255,229), white(255,255,255);
    cv::Vec3b magenta2(102,102,255), cyan2(255,255,102), lgreen2(102,255,178),grey(160,160,160);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            // show classification result
            cv::Mat sampleMat = (cv::Mat_<float>(1,2) << img.at<cv::Vec3b>(i,j)[index1_],
                                                 img.at<cv::Vec3b>(i,j)[index2_]);
            float response = SVM.predict(sampleMat);
            if (response == 1)
                img.at<cv::Vec3b>(i,j)  = magenta2;
            else if (response == 2)
                img.at<cv::Vec3b>(i,j)  = grey;
            else if (response ==3)
                img.at<cv::Vec3b>(i,j)  = lgreen2;
            else
                img.at<cv::Vec3b>(i,j)  = white;

        }
    }
    m_nanoSec = m_timer.nsecsElapsed();
    qDebug()<<"SVM testing time(ns):"<<m_nanoSec;

    /// Show the decision regions given by the SVM
    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < 256; ++j)
        {
            // show svm result
            cv::Mat sampleMat = (cv::Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                svm_img.at<cv::Vec3b>(i,j)  = magenta;
            else if (response == 2)
                svm_img.at<cv::Vec3b>(i,j)  = grey;
            else if (response ==3)
                svm_img.at<cv::Vec3b>(i,j)  = lgreen;
            else
                svm_img.at<cv::Vec3b>(i,j)  = white;
        }
    }

    /// Show the training data
    int thickness = -1;
    int lineType = 8;
    for(int i=0; i<size; i++)
    {
        cv::Point pt(trainingData[i][0],trainingData[i][1]);
        if(1 == labels[i])
            cv::circle( svm_img, pt, 2, cv::Scalar(  255,   0,   0), thickness, lineType);
        else if(2 == labels[i])
            cv::circle( svm_img, pt, 2, cv::Scalar(  0,   255,   0), thickness, lineType);
        else if(3 == labels[i])
            cv::circle( svm_img, pt, 2, cv::Scalar(  0,   0,   255), thickness, lineType);
        else
            cv::circle( svm_img, pt, 2, cv::Scalar(  0,   255/labels[i], 10), thickness, lineType);
    }

#if 0
    // Show support vectors
    thickness = 1;
    lineType = 8;
    int c     = SVM.get_support_vector_count();
    cout<<c<<endl;
    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        cv::circle( svm_img,  cv::Point( (int) v[0], (int) v[1]), 3,  cv::Scalar(255, 229, 204), thickness, lineType);
    }
#endif

    cv::imshow("SVM Result1", svm_img); // show it to the user
    cv::imwrite("svm_img.png", svm_img);
    cv::imwrite("svm_result.png", img);

}

float multi_class::error_rate()         // FIXME: stupid test
{
    unsigned int wrong=0;
    for(int i=0;i<pts_vec_.size();i++)
    {
        unsigned int type = classify(pts_vec_.at(i));
        if(type != label_vec_.at(i))
        {
            wrong++;
        }
        //cout<<"true value:"<<label_vec_.at(i)<<" test value:"<<type<<endl;
    }
    //cout<<"error rate:"<<(float)wrong/pts_vec_.size()<<endl;
    return (float)wrong/pts_vec_.size();
}
unsigned int multi_class::index1() const
{
    return index1_;
}

void multi_class::setIndex1(unsigned int index1)
{
    index1_ = index1;
}
unsigned int multi_class::index2() const
{
    return index2_;
}

void multi_class::setIndex2(unsigned int index2)
{
    index2_ = index2;
}

vector<cv::Vec2f> multi_class::get_pts_vec() const
{
    return pts_vec_;
}





