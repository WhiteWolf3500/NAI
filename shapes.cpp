#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include <math.h>
#include <string.h>

int thresh = 150;
const char* wndname = "Shapes";


static void findShapes( const Mat& image, vector<Vec3f> circles, vector<vector<Point> >& tetrahedrons)
{
    tetrahedrons.clear();
    circles.clear();

    vector<vector<Point> > contours;

    Mat srcgray, gray;
    cvtColor(image,srcgray,CV_BGR2GRAY);

        Canny(srcgray, gray, 0, thresh, 5, true);
        dilate(gray, gray, Mat(), Point(-1,-1));     
            
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
            
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    tetrahedrons.push_back(approx);
                }
            }

    HoughCircles( srcgray, circles, CV_HOUGH_GRADIENT, 1, srcgray.rows/8, 200, 50, 0, 0);
}

static void drawShapes( Mat& image, vector<vector<Point> >& shapes, string name)
{

    for( size_t i = 1; i < shapes.size(); i += 2 )
    {
        Point* p = &shapes[i][0];
        int n = (int)shapes[i].size();
        polylines(image, &p, &n, 1, true, Scalar(255,255,0), 3, LINE_AA);

        putText( image, name, cvPoint(0,0), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
    }
    imshow(wndname, image);
}

static void drawCircles( const Mat& image, vector<Vec3f> &circles)
{
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( image, center, radius, Scalar(255,255,0), 3, 8, 0 );
        putText( image, "circle", center, FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
   }
}

int main(int argc, char** argv)
{

    static const char* names[] = { "data/pic1.jpg", 0 };

    if( argc > 1)
    {
     names[0] =  argv[1];
     names[1] =  0;
    }

    namedWindow( wndname, 1 );

    vector<vector<Point> > tetrahedrons;
    vector<Vec3f> circles;

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findShapes(image, circles, tetrahedrons);
        drawShapes(image, tetrahedrons, "tetrahedrons");
        drawCircles(image, circles);

        char c = (char)waitKey();
        if( c == 27 )
            break;
    }

    // Mat srcgray, gray;
    // cvtColor(image,srcgray,CV_BGR2GRAY);
    // Canny(srcgray, gray, 0, thresh, 5, true);
    // imshow(wndname, gray);

    return 0;
}