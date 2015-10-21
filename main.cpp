#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* function prototypes */
void detect(cv::Mat &);
void detectFaces(std::vector<cv::Rect> &, cv::Mat const &, cv::CascadeClassifier &);
void detectEyes(cv::Rect &, cv::Mat &, cv::Mat const &, cv::CascadeClassifier &);
void detectSmile(cv::Rect &, cv::Mat &, cv::Mat const &, cv::CascadeClassifier &);

/** @function main */
int main()
{
    cv::VideoCapture capture(0);
    cv::Mat frame;

    if(!capture.isOpened())
    {
        printf("--(!)Error opening video capture\n");
        return -1;
    }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        detect(frame);

        if((char) cv::waitKey(10) == 27) break; // escape
    }

    return 0;
}

void detect(cv::Mat & frame)
{
    std::string const face_cascade_name = "haarcascade_frontalface_alt.xml";
    std::string const eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
    std::string const smile_cascade_name = "haarcascade_smile.xml";

    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;
    cv::CascadeClassifier smile_cascade;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return; };
    if( !smile_cascade.load( smile_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return; };

    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    std::vector<cv::Rect> faces;

    detectFaces(faces, frame_gray, face_cascade);

    for (size_t i = 0; i < faces.size(); i++)
    {
        detectEyes(faces[i], frame, frame_gray, eyes_cascade);
        detectSmile(faces[i], frame, frame_gray, smile_cascade);
    }

    cv::imshow("Cam Preview", frame);
}

void detectFaces(std::vector<cv::Rect> & faces, cv::Mat const & frame, cv::CascadeClassifier & classifier)
{
    classifier.detectMultiScale(frame, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
}

void detectEyes(cv::Rect & face, cv::Mat & frame, cv::Mat const & frame_gray, cv::CascadeClassifier & classifier)
{
    std::vector<cv::Rect> eyes;

    cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
    cv::ellipse(frame, center, cv::Size(face.width / 2, face.height / 2 ), 0, 0, 360, cv::Scalar(0, 255, 255), 4, 8, 0);

    cv::Mat faceROI = frame_gray(face);

    classifier.detectMultiScale(faceROI, eyes, 1.1, 20, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (size_t i = 0; i < eyes.size(); i++)
    {
        cv::Point eye_center(face.x + eyes[i].x + eyes[i].width / 2, face.y + eyes[i].y + eyes[i].height / 2);

        int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25 );

        cv::circle(frame, eye_center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
    }
}

void detectSmile(cv::Rect & face, cv::Mat & frame, cv::Mat const & frame_gray, cv::CascadeClassifier & classifier)
{
    std::vector<cv::Rect> smile;

    classifier.detectMultiScale(frame_gray, smile, 1.1, 125, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for(size_t j = 0; j < smile.size(); ++j) {
        cv::Point smileCenter (smile[j].x + smile[j].width / 2, smile[j].y + smile[j].height / 2);

        if(smileCenter.inside(face) && smileCenter.y >= face.y + face.height / 2) {
            cv::ellipse(frame, smileCenter, cv::Size(smile[j].width / 2, smile[j].height / 2), 0, 0, 360, cv::Scalar(0, 255, 0), 4, 8, 0);
        }
    }
}