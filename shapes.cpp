#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include <math.h>
#include <string.h>

int thresh = 120;
const char* wndname = "Shapes";


static void findShapes( const Mat& image,vector<vector<Point> >& tetrahedrons)
{
    tetrahedrons.clear();

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
}

static void drawShapes( Mat& image, vector<vector<Point> >& shapes, string name)
{

    for( size_t i = 1; i < shapes.size(); i += 2 )
    {
        Point* p = &shapes[i][0];
        int n = (int)shapes[i].size();
        polylines(image, &p, &n, 1, true, Scalar(255,255,0), 3, LINE_AA);
    }
    imshow(wndname, image);
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

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findShapes(image,tetrahedrons);
        drawShapes(image, tetrahedrons, "tetrahedrons");

        char c = (char)waitKey();
        if( c == 27 )
            break;
    }

    return 0;
}