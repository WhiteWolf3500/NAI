#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

#include <iostream>
#include <math.h>
#include <string.h>

int thresh = 100;
const char* wndname = "Shapes";

static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static void findShapes( const Mat& image, vector<Vec3f> circles, vector<vector<Point> >& triangles, vector<vector<Point> >& tetrahedrons, vector<vector<Point> >& pentagons, vector<vector<Point> >&  rectangles )
{
    tetrahedrons.clear();
    circles.clear();

    vector<vector<Point> > contours;

    Mat srcgray, gray;
    cvtColor(image,srcgray,CV_BGR2GRAY);
    GaussianBlur(srcgray,srcgray,Size(9, 9), 2, 2 );

        Canny(srcgray, gray, 0, thresh, 5, true);
        dilate(gray, gray, Mat(), Point(-1,-1));     
            
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
            
                if( approx.size() == 3 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    triangles.push_back(approx);
                }
                else if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    if( maxCosine < 0.1 )
                        rectangles.push_back(approx);
                    else
                        tetrahedrons.push_back(approx);
                }
                else if( approx.size() == 5 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    pentagons.push_back(approx);
                }
            }

    // Canny(srcgray, gray, 0, thresh, 5, true);
    HoughCircles( srcgray, circles, CV_HOUGH_GRADIENT, 1, srcgray.rows/8, 200, 50, 0, 0);
}

static void drawShapes( Mat& image, vector<vector<Point> >& shapes, string name, int r, int g, int b)
{

    for( size_t i = 1; i < shapes.size(); i += 2 )
    {
        Point* p = &shapes[i][0];
        int n = (int)shapes[i].size();
        polylines(image, &p, &n, 1, true, Scalar(r,g,b), 3, LINE_AA);

        putText( image, name, p[1], FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
    }
    imshow(wndname, image);
}

static void drawCircles( const Mat& image, vector<Vec3f> &circles, int r, int g, int b)
{
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( image, center, radius, Scalar(r,g,b), 3, 8, 0 );
        putText( image, "circle", center, FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
   }
    imshow(wndname, image);
}

static void counting( const Mat& image, vector<Vec3f> &circles, vector<vector<Point> >& triangles, vector<vector<Point> >& tetrahedrons, vector<vector<Point> >& pentagons,vector<vector<Point> >& rectangles )
{
    int x = 10, y = 20;

    if(circles.size() > 0)
    {
        String label = "Circles: " + to_string(circles.size());
        putText( image, label, cvPoint(x,y), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
        y +=15;
    }
    
    if(triangles.size() > 0)
    {
        String label = "Triangles: " + to_string(triangles.size()/2);
        putText( image, label, cvPoint(x,y), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
        y +=15;
    }
    
    if(pentagons.size() > 1)
    {
        String label = "Pentagons: " + to_string(pentagons.size()/2);
        putText( image, label, cvPoint(x,y), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
        y +=15;
    }

    if(tetrahedrons.size() > 1 )
    {
        String label = "Tetrahedrons: " + to_string(tetrahedrons.size()/2);
        putText( image, label, cvPoint(x,y), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
        y +=15;
    }

    if(rectangles.size() > 1 )
    {
        String label = "Rectangles: " + to_string(rectangles.size()/2);
        putText( image, label, cvPoint(x,y), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1, 1, false);
        y +=15;
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

    vector<vector<Point> > triangles;
    vector<vector<Point> > tetrahedrons;
    vector<vector<Point> > rectangles;
    vector<vector<Point> > pentagons;
    vector<Vec3f> circles;

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findShapes(image, circles, triangles, tetrahedrons, pentagons, rectangles);
        drawCircles(image, circles, 0, 0, 0);
        drawShapes(image, triangles, "triangles", 255, 255, 0);
        drawShapes(image, tetrahedrons, "tetrahedrons", 0, 255, 255);
        drawShapes(image, rectangles, "rectangles", 0, 255, 255);
        drawShapes(image, pentagons, "pentagons", 255, 0, 255);
        counting(image, circles, triangles, tetrahedrons, pentagons, rectangles);



        // Mat srcgray, gray;
        // cvtColor(image,srcgray,CV_BGR2GRAY);
        // Canny(srcgray, gray, 0, thresh, 5, true);
        // imshow(wndname, gray);


        char c = (char)waitKey();
        if( c == 27 )
            break;
    }


    return 0;
}