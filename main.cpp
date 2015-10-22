/* libraries */
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* function prototypes */
void detect(cv::Mat &, cv::CascadeClassifier &, cv::CascadeClassifier &, cv::CascadeClassifier &);
void detectFaces(std::vector<cv::Rect> &, cv::Mat const &, cv::CascadeClassifier &);
void detectEyes(cv::Rect const &, cv::Mat &, cv::Mat const &, cv::CascadeClassifier &);
void detectSmile(cv::Rect const &, cv::Mat &, cv::Mat const &, cv::CascadeClassifier &);

/*
 * constants
 */
std::string const static FACE_CASCADE_NAME = "haarcascade_frontalface_alt.xml";
std::string const static EYES_CASCADE_NAME = "haarcascade_eye_tree_eyeglasses.xml";
std::string const static SMILE_CASCADE_NAME = "haarcascade_smile.xml";

/**
 *  global variables
 */
cv::CascadeClassifier static face_cascade;
cv::CascadeClassifier static eyes_cascade;
cv::CascadeClassifier static smile_cascade;

/**
 * \brief main function serving as entry point to program
 *
 * @return 0 if programm was executed correctly
 */
int main()
{
    cv::VideoCapture capture(0);
    cv::Mat frame;

    if(!face_cascade.load(FACE_CASCADE_NAME)) { std::cout << "--(!)Error loading face cascade" << std::endl; return -1; };
    if(!eyes_cascade.load(EYES_CASCADE_NAME)) { std::cout << "--(!)Error loading eyes cascade" << std::endl; return -1; };
    if(!smile_cascade.load(SMILE_CASCADE_NAME)) { std::cout << "--(!)Error loading smile cascade" << std::endl; return -1; };

    if(!capture.isOpened())
    {
        std::cout << "--(!)Error opening video capture" << std::endl;
        return -1;
    }

    while (capture.read(frame))
    {
        if( frame.empty() )
        {
            std::cout << " --(!) No captured frame -- Break!" << std::endl;
            break;
        }

        detect(frame, face_cascade, eyes_cascade, smile_cascade);

        cv::imshow("Cam Preview", frame);

        if((char) cv::waitKey(10) == 27) break; // escape
    }

    return 0;
}

/**
 * \brief wrapper function for facial recognition
 *
 * @param[out] frame holds image data
 * @param[in] face_cascade facial recognition
 * @param[in] eyes_cascade eyes detection
 * @param[in] smile_cascade smile detection
 *
 * Highlights all detected features, face, eyes, smile in the camera image
 *
 * @see main()
 */
void detect(cv::Mat & frame, cv::CascadeClassifier & face_cascade, cv::CascadeClassifier & eyes_cascade, cv::CascadeClassifier & smile_cascade)
{
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    std::vector<cv::Rect> faces;

    detectFaces(faces, frame_gray, face_cascade);

    for(std::vector<cv::Rect>::iterator it = faces.begin(); it != faces.end(); ++it)
    {
        detectEyes(* it, frame, frame_gray, eyes_cascade);
        detectSmile(* it, frame, frame_gray, smile_cascade);
    }
}

/**
 * \brief detects faces in image
 *
 * @param[out] faces a data structure holding data of found faces
 * @param[in] frame holds the current camera image
 * @param[in] classifier detects faces in images
 *
 * retrieves all faces in an image and saves them to a given data structure
 *
 * @see detect()
 */
void detectFaces(std::vector<cv::Rect> & faces, cv::Mat const & frame, cv::CascadeClassifier & classifier)
{
    classifier.detectMultiScale(frame, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
}

/**
 * \brief detects eyes in an image
 *
 * @param[in] face represents the current face to detect eyes in
 * @param[out] frame holds the data of the current image
 * @param[in] frame_gray contains the equalized grayscale image of the current image
 * @param[in] classifier detects eyes in images
 *
 * retrieves all eyes detected in a face and highlights them in the current camera image
 *
 * @see detect()
 */
void detectEyes(cv::Rect const & face, cv::Mat & frame, cv::Mat const & frame_gray, cv::CascadeClassifier & classifier)
{
    std::vector<cv::Rect> eyes;

    cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
    cv::ellipse(frame, center, cv::Size(face.width / 2, face.height / 2 ), 0, 0, 360, cv::Scalar(0, 255, 255), 4, 8, 0);
    cv::Mat faceROI = frame_gray(face);

    classifier.detectMultiScale(faceROI, eyes, 1.1, 20, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for(std::vector<cv::Rect>::iterator it = eyes.begin(); it != eyes.end(); ++it)
    {
        cv::Point eye_center(face.x + (* it).x + (* it).width / 2, face.y + (* it).y + (* it).height / 2);

        int radius = cvRound(((* it).width + (* it).height) * 0.25);

        cv::circle(frame, eye_center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
    }
}

/**
 * \brief detects smile in an image
 *
 * @param[in] face represents the current face to detect smiles in
 * @param[out] frame holds the data of the current image
 * @param[in] frame_gray contains the equalized grayscale image of the current image
 * @param[in] classifier detects smiles in images
 *
 * retrieves all smiles detected in a face and highlights them in the current camera image
 *
 * @see detect()
 */
void detectSmile(cv::Rect const & face, cv::Mat & frame, cv::Mat const & frame_gray, cv::CascadeClassifier & classifier)
{
    std::vector<cv::Rect> smile;

    cv::Mat faceROI = frame_gray(face);

    classifier.detectMultiScale(faceROI, smile, 1.1, 125, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for(std::vector<cv::Rect>::iterator it = smile.begin(); it != smile.end(); ++it)
    {
        cv::Point smileCenter(face.x + (* it).x + (* it).width / 2, face.y + (* it).y + (* it).height / 2);
        cv::ellipse(frame, smileCenter, cv::Size((* it).width / 2, (* it).height / 2), 0, 0, 360, cv::Scalar(0, 255, 0), 4, 8, 0);
    }
}