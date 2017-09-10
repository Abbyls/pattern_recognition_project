#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;

namespace cv_class
{
    class cv_classification
    {

    public:
        cv_classification();
        cv_classification(int buffer_size);
        ~cv_classification();

        void                        push_good_pt(cv::Vec3f &pt);
        void                        push_bad_pt(cv::Vec3f &pt);
        cv::Vec3f                   get_good_pt(int index);
        cv::Vec3f                   get_bad_pt(int index);
        vector<cv::Vec3f>           get_good_vec();
        vector<cv::Vec3f>           get_bad_vec();
        void                        clear();

        // general methods
        void                cal_line(cv::Vec2f pt1, cv::Vec2f pt2, float &A, float &B, float &C);
        cv::Vec3f           get_mid_pt(vector<cv::Vec3f>& pts);
        Eigen::Vector3f     cv2eigen_vector(cv::Vec3f& vec);

        // specific methods
        void                LDA(Eigen::Vector2f &w, float &thres);
        void                SVM(cv::Mat &img);
        void                my_own_classify(float &k, float &b, float &x0, float &y0);
        float               cal_thres(float k);
        void                test();

        unsigned int        index1() const;
        void                setIndex1(unsigned int index1);

        unsigned int        index2() const;
        void                setIndex2(unsigned int index2);

    private:
        vector<cv::Vec3f>       good_vec_;
        vector<cv::Vec3f>       bad_vec_;
        Eigen::Matrix2f         Sw_;
        unsigned int            index1_;
        unsigned int            index2_;

    };

}

#endif // CLASSIFICATION_H
