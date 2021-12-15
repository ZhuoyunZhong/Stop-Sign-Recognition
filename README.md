# Stop Sign Recognition

RBE 549 Computer Vision Final Project.

## Dataset

We are using a dataset collected by UCSD Self-driving cars team [LISA](http://cvrr-nas.ucsd.edu//LISA/lisa-traffic-sign-dataset.html) and stop sign images collected [from the internet](https://github.com/mbasilyan/Stop-Sign-Detection/tree/master/Stop%20Sign%20Dataset). We made some modifications for better testing our algorithms. Please refer to the README in dataset folder for more details.

A sample of the Online dataset and LISA dataset

![](demo/sample.jpg)

## Run

To run the dataset test,

Method 1: Feature matching

`python3 feature_matching.py`

Method 2: Haar cascaded classifier

`python3 haar_cascade.py`

Method 3: HSV-SVM classifier

`python3 hsv_svm.py `



Recognition result

![](demo/result.jpg)

## Algorithms

##### Feature matching

Matching model and image

![](demo/feature/image_match.jpg)

Matching model and ROI

![](demo/feature/roi_match.jpg)

##### Haar Cascade

Haar Feature

![](demo/haar/haar.jpg)

##### HSV-SVM 

HSV ROI Extraction

![](demo/extraction/process.jpg)

extraction results

![](demo/extraction/4.1.jpg) ![](demo/extraction/4.2.jpg) ![](demo/extraction/4.3.jpg)

SVM features

![](demo/svm/feature.jpg)