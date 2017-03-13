**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/car.png
[image1]: ./output_images/notcar.png
[image2]: ./output_images/spatial.PNG
[image3]: ./output_images/histogram_test_img.jpg
[image4]: ./output_images/example_hog.jpg
[image5]: ./output_images/hm1.png
[image6]: ./output_images/hm2.png
[image7]: ./output_images/hm3.png
[image8]: ./output_images/heatmap1.png
[image9]: ./output_images/detected2.png
[image10]: ./output_images/hogsubsampling.PNG
[video1]: ./project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

####2. Files provided:

- main.py - initialization, training, detection
- VehicleDetector.py - Class for holding all parameters regarding training and detection
- functions.py - helper class with utilization functions
- calibration_data.p - camera undistortion pickle file for fast calibration
- detector_pickle.p - trained vehicle detector object for fast loading
 
[Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained hog_classify() in lines 106 through 197 of the file called `VehicleDetector.py`).  

I started by loading previous calibration data from the Advanced Lane Lines project (Assuming we used the same camera, line 181)
Then i started reading in all the `vehicle` and `non-vehicle` images on `main.py`, see lines 87-106. 

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car:

![alt text][image0]

Not a car:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientation=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2] TODO

####2. Explain how you settled on your final choice of HOG parameters.

Parameters where chosen with a trial and error approach approximating the final values. I started from the ones used in the lectures and decided to finally
go with those below, resulting in an accuracy of 99,51% on 1000 samples:

Final:
orientations: 7
pixels_per_cell: 2,2
cells_per_block: 8,8


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As this is my first vehicle detector i stayed close to the lectures and tried to implement the hog classifier 
by changing the color space to `YCrCb` and extracting hog features of all 3 channels. Furthermore i added spatial and color histogram features.
Spatial binning is an excellent function to preserve information about the object but with very low pixel rates.
Information about the color intensity and possible clusters can be gathered by calculation the color histogram.
A `StandardScaler` was used to remove the mean and scaling to unit variance. 

Example of a color histogram:

![alt text][image3]

Example of spatial binning:

![alt text][image2]

Example of hog features 8x8 pixels per cell and 2x2 cells per block:

![alt text][image4]


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I chose to try it with the hog subsampling search introduced in the lessons. It seemed to be reasonably good and easy to implement as well it was#
also stated that it's more efficient than a bare sliding window search. 
What makes it superior is the fact that hog features need to be extract once and then can be sub-sampled to get all of its overlaying windows.
Each window is defined by a scaling factor which makes it easy to run it several times with different scaling to detect object from different sizes.
Instead of overlapping cells_per_step is used. (`functions.py` line 118)

![alt text][image10]

I chose one window with ystart 390 and ystop of 660.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


Ultimately I searched on one window with 1.5 scale using YCrCb 3-channel HOG features plus spatially binned and histograms of color in the feature vector, which provided a nice result.

See section 3 for images of spatial binning, color histogram and hog features.

By using higher threshold for heatmaps i could eliminate some more false positives.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` (line 246 in file `VehicleDetector.py` )to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.(`draw_labeled_bboxes()` lines 32 through 49 in `functions.py` )  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the resulting bounding boxes

### Here are 3 mostly consecutive frames and their corresponding heatmaps and the resulting bounding boxes in pink:

![alt text][image5]
![alt text][image6]
![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on a sample frame for a single detection:
![alt text][image8]

The heatmaps have been stored in a queue of length 10. based on those a sum is calculated over all and used as a threshold for filtering out 
false positives. (`calc_avg_heat` in `VehicleDetector.py` lines 293 through 306)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For better detection i could have use more windows on different scales to also detect cars which or on the upper and of the detection area.
It would also be good to provide more stable and better surrounding bounding boxes by calcultation the average bounding box of a detected vehicle.
Some changes in the light conditions also makes the detection more difficult which i did not address yet.

