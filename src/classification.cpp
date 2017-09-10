#include "classification.h"
#include <iostream>
#include <QElapsedTimer>
#include <QDebug>

using namespace cv_class;
using namespace cv;

qint64          nanoSec=0;
QElapsedTimer   timer;

cv_classification::cv_classification():index1_(0),index2_(1)
{
    good_vec_.reserve(100);
    bad_vec_.reserve(100);
    good_vec_.clear();
    bad_vec_.clear();
}

cv_classification::cv_classification(int buffer_size):index1_(0),index2_(1)
{
    good_vec_.reserve(buffer_size);
    bad_vec_.reserve(buffer_size);
    good_vec_.clear();
    bad_vec_.clear();

}

cv_classification::~cv_classification()
{}

void cv_classification::push_good_pt(cv::Vec3f &pt)
{
    good_vec_.push_back(pt);
}

void cv_classification::push_bad_pt(cv::Vec3f &pt)
{
    bad_vec_.push_back(pt);
}

cv::Vec3f cv_classification::get_good_pt(int index)
{
    return good_vec_.at(index);
}

cv::Vec3f cv_classification::get_bad_pt(int index)
{
    return bad_vec_.at(index);
}

vector<cv::Vec3f> cv_classification::get_good_vec()
{
    return good_vec_;
}

vector<cv::Vec3f> cv_classification::get_bad_vec()
{
    return bad_vec_;
}

void cv_classification::clear()
{
    good_vec_.clear();
    bad_vec_.clear();
}

void cv_classification::my_own_classify(float &k, float &b, float &x0, float &y0)
{
    cv::Vec3f   mid1(0,0,0), mid2(0,0,0), mid_pt(0,0,0);
    float       x=0 ,y=0;

    mid1 = get_mid_pt(good_vec_);
    mid2 = get_mid_pt(bad_vec_);
    mid_pt = (mid1 + mid2) / 2.0;
    x0 = mid_pt.val[index1_];
    y0 = mid_pt.val[index2_];

    float sum_g1=0, sum_g2=0;
    for(cv::Vec3f n : good_vec_)      // red: n.val[0]; green: n.val[1];  blue: n.val[2]
    {
        x = n.val[index1_];
        y = n.val[index2_];
        sum_g1 += (y-y0)*(y-y0);
        sum_g2 += (x-x0)*(y-y0);
    }

    float sum_b1=0, sum_b2=0;
    for(cv::Vec3f n : bad_vec_)      // red: n.val[0]; green: n.val[1];  blue: n.val[2]
    {
        x = n.val[index1_];
        y = n.val[index2_];
        sum_b1 += (y-y0)*(y-y0);
        sum_b2 += (x-x0)*(y-y0);
    }
    k = (sum_g1 - sum_b1) / (sum_g2 - sum_b2);
    b = y0-k*x0;
}

// line function: Ax + By + C = 0
void cv_classification::cal_line(Vec2f pt1, Vec2f pt2, float &A, float &B, float &C)
{
    float x1 = pt1[0], x2 = pt2[0];
    float y1 = pt1[1], y2 = pt2[1];

    if(x1 != x2)
    {
        B = 1;
        A = -(y1-y2)/(x1-x2);
        C = -B*y1 - A*x1;
    }
    else
    {
        B = 0;
        A = 1;
        C = -x1;
    }
//    cout<<"line equation: "<<A<<"x+"<<B<<"y+"<<C<<"=0"<<endl;
}

float cv_classification::cal_thres(float k)   // k is a parameter
{
    float thres=-1;
    if(k>=0.0 && k<=1.0)
    {
        float good_percent=0, bad_percent=0;
        for(cv::Vec3f n : good_vec_)      // red: n.val[0]; green: n.val[1];  blue: n.val[2]
            good_percent += n.val[0] / (n.val[0] + n.val[1] + n.val[2]);
        good_percent = good_percent / (float)good_vec_.size();

        for(cv::Vec3f n : bad_vec_)      // red: n.val[0]; green: n.val[1];  blue: n.val[2]
            bad_percent += n.val[0] / (n.val[0] + n.val[1] + n.val[2]);
        bad_percent = bad_percent / (float)bad_vec_.size();

        thres = k * good_percent + (1-k) * bad_percent;
        return thres;
    }
    else
        return -1;
}

Eigen::Vector3f cv_classification::cv2eigen_vector(cv::Vec3f &vec)
{
    Eigen::Vector3f e_vec;
    e_vec << vec.val[0], vec.val[1], vec.val[2];
    return e_vec;
}

cv::Vec3f cv_classification::get_mid_pt(vector<cv::Vec3f> &pts)
{
    cv::Vec3f mid;
    for(cv::Vec3f n : pts)      // red: n.val[0]; green: n.val[1];  blue: n.val[2]
        mid += n;
    mid = mid / (float)pts.size();
    return mid;
}

void cv_classification::LDA(Eigen::Vector2f &w, float &thres)
{
    cv::Vec3f c_m1 = get_mid_pt(good_vec_);
    cv::Vec3f c_m2 = get_mid_pt(bad_vec_);
    Eigen::Vector2f e_m1(c_m1.val[index1_],c_m1.val[index2_]);
    Eigen::Vector2f e_m2(c_m2.val[index1_],c_m2.val[index2_]);
    Eigen::Matrix2f s1, s2;
    s1 << 0,0,
          0,0;
    s2 << 0,0,
          0,0;

    timer.start();

    for(cv::Vec3f n : good_vec_)
    {
        Eigen::Vector2f e_pt(n.val[index1_],n.val[index2_]);
        s1 += (e_pt - e_m1) * ((e_pt - e_m1).transpose());
    }
    for(cv::Vec3f n : bad_vec_)
    {
        Eigen::Vector2f e_pt(n.val[index1_],n.val[index2_]);
        s2 += (e_pt - e_m2) * ((e_pt - e_m2).transpose());
    }
    Sw_ = s1 + s2;

    w = Sw_.inverse() * (e_m1 - e_m2);
    Eigen::VectorXf thres_t(1,1);
    thres_t = (w.transpose() * e_m1 + w.transpose() * e_m2 )/2.0;
    thres = thres_t(0);

    nanoSec = timer.nsecsElapsed();
    qDebug()<<"LDA traning uses time(ns):"<<nanoSec;

    cout<<"sw:"<<endl<<Sw_<<endl;
    cout<<"w:"<<w<<endl;
    cout<<"thres:"<<thres<<endl;
}

void cv_classification::SVM(cv::Mat &img)
{
    cv::Mat svm_img = cv::Mat::zeros(256, 256, CV_8UC3);

    /// Set up training data
    int     g_size = good_vec_.size(),
            b_size = bad_vec_.size(),
            size = g_size + b_size;
    float   labels[size], trainingData[size][2];

    for(int i=0; i<size; i++)
    {
        if(i<g_size)
        {
            labels[i] = 1;
            trainingData[i][0] = good_vec_.at(i).val[index1_];
            trainingData[i][1] = good_vec_.at(i).val[index2_];
        }
        else
        {
            labels[i] = -1;
            trainingData[i][0] = bad_vec_.at(i-g_size).val[index1_];
            trainingData[i][1] = bad_vec_.at(i-g_size).val[index2_];
        }
    }
    cv::Mat labelsMat(size, 1, CV_32FC1, labels);
    cv::Mat trainingDataMat(size, 2, CV_32FC1, trainingData);

    /// Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    /// Train the SVM
    timer.start();

    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

    nanoSec = timer.nsecsElapsed();
    qDebug()<<"SVM training time(ns):"<<nanoSec;

    /// Test image response
    timer.start();

    cv::Vec3b red(0,0,255), white (255,255,255);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            // show classification result
            cv::Mat sampleMat = (cv::Mat_<float>(1,2) << img.at<cv::Vec3b>(i,j)[index1_],
                                                 img.at<cv::Vec3b>(i,j)[index2_]);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                img.at<cv::Vec3b>(i,j)  = red;
            else if (response == -1)
                img.at<cv::Vec3b>(i,j)  = white;
        }
    }
    nanoSec = timer.nsecsElapsed();
    qDebug()<<"SVM testing time(ns):"<<nanoSec;

    /// Show the decision regions given by the SVM
    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < 256; ++j)
        {
            // show svm result
            cv::Mat sampleMat = (cv::Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                svm_img.at<cv::Vec3b>(i,j)  = red;
            else if (response == -1)
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
            cv::circle( svm_img, pt, 2, cv::Scalar(  0,   255,   0), thickness, lineType);
        else
            cv::circle( svm_img, pt, 2, cv::Scalar(  0,   0,   0), thickness, lineType);
    }

#if 0
    // Show support vectors
    thickness = 2;
    lineType = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        cv::circle( svm_img,  cv::Point( (int) v[0], (int) v[1]),   6,  cv::Scalar(255, 0, 0), thickness, lineType);
    }
#endif
    //cv::imwrite("../result.png", svm_img);        // save the image
    cv::imshow("SVM Result", svm_img); // show it to the user
    cv::imwrite("svm_img.png", svm_img);        // save the image
    cv::imwrite("svm_result.png", img);        // save the image
}

void cv_classification::test()
{
#if 0
    // Data for visual representation
       int width = 512, height = 512;
       Mat image = Mat::zeros(height, width, CV_8UC3);

       // Set up training data
       float labels[4] = {1.0, -1.0, -1.0, -1.0};
       Mat labelsMat(4, 1, CV_32FC1, labels);

       float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
       Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

       // Set up SVM's parameters
       CvSVMParams params;
       params.svm_type    = CvSVM::C_SVC;
       params.kernel_type = CvSVM::LINEAR;
       params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

       // Train the SVM
       CvSVM SVM;
       SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

       Vec3b green(0,255,0), blue (255,0,0);
       // Show the decision regions given by the SVM
       for (int i = 0; i < image.rows; ++i)
           for (int j = 0; j < image.cols; ++j)
           {
               Mat sampleMat = (Mat_<float>(1,2) << j,i);
               float response = SVM.predict(sampleMat);

               if (response == 1)
                   image.at<Vec3b>(i,j)  = green;
               else if (response == -1)
                    image.at<Vec3b>(i,j)  = blue;
           }

       // Show the training data
       int thickness = -1;
       int lineType = 8;
       circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
       circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
       circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
       circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

       // Show support vectors
       thickness = 2;
       lineType  = 8;
       int c     = SVM.get_support_vector_count();

       for (int i = 0; i < c; ++i)
       {
           const float* v = SVM.get_support_vector(i);
           circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
       }

       imwrite("result.png", image);        // save the image

       imshow("SVM Simple Example", image); // show it to the user
       waitKey(0);
#endif
}
unsigned int cv_classification::index1() const
{
    return index1_;
}

void cv_classification::setIndex1(unsigned int index1)
{
    index1_ = index1;
}

unsigned int cv_classification::index2() const
{
    return index2_;
}

void cv_classification::setIndex2(unsigned int index2)
{
    index2_ = index2;
}



/******* OpenCV Code Reference **************/
//    image = cv::imread("../img/2.jpg");
//    if(!image.data)
//        return;

//    cv::namedWindow("My Image");
//    cv::imshow("My Image", image);
//    cv::waitKey(1000);
//      cv::namedWindow("Original Image");
//      cv::imshow("Original Image", image_);

    //QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"),".",
    //                                                tr("Image Files(*.png *.jpg *.jpeg *.bmp)"));
    //image_ = cv::imread(fileName.toStdString().data());
