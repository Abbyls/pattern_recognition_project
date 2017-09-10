#ifndef MULTI_CLASS_H
#define MULTI_CLASS_H

#include "classification.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <eigen3/Eigen/Dense>
#include <QTime>

using namespace std;
using namespace cv_class;

class multi_class : public cv_classification
{
public:
    multi_class();
    ~multi_class();

    void                        push_pts(cv::Vec2f &pt);
    void                        push_manual(cv::Vec2f &pt);
    unsigned int                next_label();
    unsigned int                get_label();
    int                         rand_num(int max);
    void                        clear();

    void                        cal_mid_pt();
    void                        cal_manual_line();
    void                        train_manual();
    unsigned int                classify(cv::Vec2f &pt);
    void                        SVM(cv::Mat &img);
    vector<cv::Vec3f>           line_vec_;
    vector<cv::Vec2f>           manual_vec_;

    float                       error_rate();

    unsigned int                index1() const;
    void                        setIndex1(unsigned int index1);
    unsigned int                index2() const;
    void                        setIndex2(unsigned int index2);

    vector<cv::Vec2f>           get_pts_vec() const;

private:
    vector<cv::Vec2f>           pts_vec_;
    vector<cv::Vec2f>           mid_vec_;
    vector<unsigned int>        label_vec_;
    unsigned int                label_;
    bool                        preprocess_;
    unsigned int                index1_;
    unsigned int                index2_;
};

#endif // MULTI_CLASS_H
