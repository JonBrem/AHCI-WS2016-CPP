#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

const char* cascade_name_face = "haarcascade_frontalface_default.xml";
const char* cascade_name_eyes = "haarcascade_eye_tree_eyeglasses.xml";
const char* cascade_name_smile = "haarcascade_smile.xml";

void detectAndDisplay(cv::Mat frame);
cv::Rect detectFace(cv::Mat frame);

cv::CascadeClassifier faceCascade;
cv::CascadeClassifier smileCascade;
cv::CascadeClassifier eyesCascade;

int main() {
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    if(!faceCascade.load(cascade_name_face)) {
        printf("Error loading face cascade");
        return -1;
    }

    if(!smileCascade.load(cascade_name_smile)) {
        printf("Error loading smile cascade");
        return -1;
    }

    if(!eyesCascade.load(cascade_name_eyes)) {
        printf("Error loading eye cascade");
        return -1;
    }

    cv::namedWindow("Cam",1);
    for(;;)
    {
        cv::Mat frame;
        cap >> frame; // get a new frame from camera

        detectAndDisplay(frame);

        if(cv::waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

void detectAndDisplay(cv::Mat frame) {
    cv::Rect face(-1, -1, -1, -1);
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    face = detectFace(frame_gray);
    if(face.x == -1) {
        imshow("Cam", frame);
        return;
    }

    cv::Point center (face.x + face.width / 2, face.y + face.height / 2);
    cv::ellipse(frame, center, cv::Size(face.width / 2, face.height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

    // SMILE DETECTION @TODO decompose
    std::vector<cv::Rect> smile;
    cv::Mat faceMat = frame_gray(face);

    smileCascade.detectMultiScale(frame_gray, smile, 1.1, 200, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for(size_t j = 0; j < smile.size(); ++j) {
        cv::Point smileCenter (smile[j].x + smile[j].width / 2, smile[j].y + smile[j].height / 2);

        if(smileCenter.inside(face) && smileCenter.y >= face.y + face.height / 2) {
            cv::ellipse(frame, smileCenter, cv::Size(smile[j].width / 2, smile[j].height / 2), 0, 0, 360,
                        cv::Scalar(255, 0, 0), 4, 8, 0);
        }
    }

    // EYE DETECTION @TODO decompose @TODO error!! crashes system
    //std::vector<cv::Rect> eyes;
    //eyesCascade.detectMultiScale(frame_gray, eyes, 1.1, 2, 0|cv::CASCADE_SCALEyw_IMAGE, cv::Size(30, 30));
    // show eyes on frame

    imshow("Cam", frame);
}

cv::Rect detectFace(cv::Mat grayFrame) {
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    int heighestSizeIndex = -1;
    int heighestSize = 0;

    for(size_t i = 0; i < faces.size(); ++i) {
        int size = faces[i].height * faces[i].width;
        if (size > heighestSize) {
            heighestSizeIndex = i;
            heighestSize = size;
        }
    }

    if(heighestSizeIndex != -1) {
        return faces[heighestSizeIndex];
    } else {
        cv::Rect r(-1, -1, -1, -1);
        return r;
    }
}