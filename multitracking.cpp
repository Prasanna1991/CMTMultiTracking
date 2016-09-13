#include "CMT.h"
#ifdef DEBUG_MODE
#	include <QDebug>
#endif

using namespace cv;
using namespace std;


int main()
{
    string fn_haar = string("../data/haarcascade_frontalface_default.xml");
    CascadeClassifier haar_cascade;

    // CMT initialization for multiple object
    CMT cmt[10];
    int objecttocreate = 10;
    for (int i=0; i<objecttocreate; i++)
    {
        cmt[i] = CMT();
    }

    int px, py;
    int cx, cy;

    cv::Mat img, im_gray, im_out;

    //open video capture
    cv::VideoCapture vf(0);

    Mat frame; //holds the current frame from the video device

    cout << "flag1"<<endl;

    int aa = 1;
    while(vf.isOpened())
    {
        vf >> frame;
        imshow("frame1", frame);
        char key = (char) waitKey(20);
        if (aa==1)
            cout << "press escape to continue detection and tracking"<<endl;
        aa=2;
        if (key!=-1)
            break;
    }

    int frame_num = 1;
    Mat gray;
    vector< Rect_<int> > faces;
    im_out = frame.clone(); //clone the current frame
    cv::cvtColor(frame, im_gray, CV_RGB2GRAY);
    if (!haar_cascade.load(fn_haar))
    {
        printf("Error loading face classifier\n");
        exit(-1);
    };
    haar_cascade.detectMultiScale(im_gray, faces);
    cout<<faces.size()<<endl;
    vector<cv::Rect>::const_iterator r;
    int ii=0;
    r=faces.begin();

    int iii=0;
    for (r=faces.begin(); r!=faces.end(); r++){
        px = r->x;
        //cout << px <<endl;
        py = r->y;
        cx=r->x + r->width;
        cy=r->y + r->height;
        //detection upto here, where the result is the corrdinates of the face as detected by the haar cascades

        cv::Point2f initTopLeft(px,py);
        cv::Point2f initBottomDown(cx,cy);
        std::cout << "learning model for bbox: " << initTopLeft << " .. " << initBottomDown << std::endl;

        if (cmt[iii].initialise(im_gray, initTopLeft, initBottomDown)==1)
            iii++;
    }
    //cmt initilazation finishes here

    cout << "succesfully initialized"<<std::endl;
    int b=1;
    //tracking process starts here
    while (vf.isOpened())
    {
        Mat freshframe;
        frame_num++;
        Mat fresh_gray, im_out_fresh;
        vf>>freshframe;

        im_out_fresh = freshframe.clone(); //clone the current frame
        cv::cvtColor(freshframe,fresh_gray, CV_RGB2GRAY);

        // tracking
        for (int ccc=0; ccc!=iii; ccc++)
            cmt[ccc].processFrame(fresh_gray);

        //draw the keypoints
        for (int kkk=0; kkk!=iii; kkk++)
        {
            for(int i = 0; i<cmt[kkk].trackedKeypoints.size(); i++)
                cv::circle(im_out_fresh, cmt[kkk].trackedKeypoints[i].first.pt, 3, cv::Scalar(255,255,255));

            // bounding box
            cv::line(im_out_fresh, cmt[kkk].topLeft, cmt[kkk].topRight, cv::Scalar(255,255,255));
            cv::line(im_out_fresh, cmt[kkk].topRight, cmt[kkk].bottomRight, cv::Scalar(255,255,255));
            cv::line(im_out_fresh, cmt[kkk].bottomRight, cmt[kkk].bottomLeft, cv::Scalar(255,255,255));
            cv::line(im_out_fresh, cmt[kkk].bottomLeft, cmt[kkk].topLeft, cv::Scalar(255,255,255));
        }
#ifdef DEBUG_MODE
        qDebug() << "trackedKeypoints";
            for(int i = 0; i<cmt.trackedKeypoints.size(); i++)
                    qDebug() << cmt.trackedKeypoints[i].first.pt.x << cmt.trackedKeypoints[i].first.pt.x << cmt.trackedKeypoints[i].second;
            qDebug() << "box";
			qDebug() << cmt.topLeft.x << cmt.topLeft.y;
			qDebug() << cmt.topRight.x << cmt.topRight.y;
			qDebug() << cmt.bottomRight.x << cmt.bottomRight.y;
			qDebug() << cmt.bottomLeft.x << cmt.bottomLeft.y;
#endif

        imshow("frame", im_out_fresh);
        char key = (char) waitKey(20);
        if (key!=-1)
            break;
    }
    return 0;
}
