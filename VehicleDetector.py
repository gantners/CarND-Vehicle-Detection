import time
import functions
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import cv2
import os
import pickle


class VehicleDetector:
    """
    Class for storing various configurations and maintaining objects over time
    """

    def __init__(self, color_space='YCrCb', orient=7, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                 spatial_size=(32, 32), hist_bins=32, spatial_feat=True, hist_feat=True, hog_feat=True, overlap=0.9,
                 hist_range=(0, 256), hog_vis=True, hog_feature_vec=True, C=0.01, ystart=400, ystop=660, scale=1.5):
        self.color_space = color_space  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient  # typically between 6 and 12
        self.pix_per_cell = pix_per_cell  # HOG pixels per cell
        self.cell_per_block = cell_per_block  # HOG cells per block
        self.hog_channel = hog_channel  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size  # Spatial binning dimensions
        self.hist_bins = hist_bins  # Number of histogram bins
        self.spatial_feat = spatial_feat  # Spatial features on or off
        self.hist_feat = hist_feat  # Histogram features on or off
        self.hog_feat = hog_feat  # HOG features on or off
        self.hist_range = hist_range
        self.hog_vis = hog_vis
        self.hog_feature_vec = hog_feature_vec

        self.overlap = overlap  # Overlapping of windows

        self.svc = None
        self.C = C  # Avoid misclassification - if too small, more false positives
        self.X_scaler = None

        self.calibration = None
        self.cal_file = 'calibration_data.p'
        self.show_heat = False
        self.show_labels = True
        self.show_boxes = True

        self.ystart = ystart
        self.ystop = ystop
        self.scale = scale
        self.heatmaps = []
        self.avg_heatmap = None

    def calibrate_camera(self):
        """
        Calibrate the camera or load calibration from a previously saved file
        :return:
        """
        do_cal = True if not os.path.exists(self.cal_file) else False
        if do_cal:
            print('Calibration for image distortion is missing')
            return False

        else:
            self.load_cal()
            print('Calibration data loaded from file', self.cal_file)
            return True

    def load_cal(self):
        """
        Load calibration from file
        :return:
        """
        with open(self.cal_file, 'rb') as file:
            cal = pickle.load(file)
        self.calibration = cal
        return cal

    def hog_classify(self, cars, notcars, color_space='YCrCb', orient=7, pix_per_cell=8, cell_per_block=2,
                     hog_channel=0, spatial_size=(32, 32), sample_size=None, hist_bins=32, spatial_feat=True,
                     hist_feat=True, hog_feat=True, hist_range=None):
        """
        Train hog classifier
        :param orient:
        :param pix_per_cell:
        :param cell_per_block:
        :param hog_channel: Can be 0, 1, 2, or "ALL"
        :return:
        :param cars:
        :param notcars:
        :param sample_size:  Reduce the sample size because HOG features are slow to compute
        :param colorspace: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        :return:
        """
        if sample_size is not None:
            cars = cars[0:sample_size]
            notcars = notcars[0:sample_size]

        t = time.time()

        car_features = functions.extract_features(cars, color_space=color_space, orient=orient,
                                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                  hog_channel=hog_channel, spatial_size=spatial_size,
                                                  hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                                  hog_feat=hog_feat, hist_range=hist_range)

        notcar_features = functions.extract_features(notcars, color_space=color_space, orient=orient,
                                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                     hog_channel=hog_channel, spatial_size=spatial_size,
                                                     hist_bins=hist_bins, spatial_feat=spatial_feat,
                                                     hist_feat=hist_feat, hog_feat=hog_feat, hist_range=hist_range)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        print(len(car_features))
        print(np.array(car_features).shape)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        draw = False
        if draw:
            car_ind = np.random.randint(0, len(cars))
            # Plot an example of raw and scaled features
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(mpimg.imread(cars[car_ind]))
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(X[car_ind])
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_X[car_ind])
            plt.title('Normalized Features')
            fig.tight_layout()

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC(C=self.C)
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 100
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
        self.svc = svc
        self.X_scaler = X_scaler
        return n_predict

    def process_frame(self, image):
        """
        Process a single image frame and do hog subsampling to find cars and improve detection by using heatmap
        :param image:
        :return:
        """

        # undistort image
        image = cv2.undistort(image, self.calibration['mtx'], self.calibration['dist'], None, self.calibration['mtx'])

        # normalize 0-1
        image = image.astype(np.float32) / 255

        draw = False

        window_img, hot_windows = functions.find_cars(image, self.ystart, self.ystop, 1.5, self.svc,
                                                      self.X_scaler,
                                                      self.orient,
                                                      self.pix_per_cell,
                                                      self.cell_per_block, self.spatial_size,
                                                      self.hist_bins, self.hist_range)

        # if windows found which possibly can be cars
        if len(hot_windows) > 0:

            print('Found {} windows'.format(len(hot_windows)))

            heat = np.zeros_like(window_img[:, :, 0]).astype(np.float)

            # Add heat to each box in box list
            heat = functions.add_heat(heat, hot_windows)

            # Apply threshold to help remove false positives
            heat = functions.apply_threshold(heat, 2)

            # add to list of previous heatmaps
            self.heatmaps.append(heat)

            # queue only latest 10 heatmaps
            if len(self.heatmaps) > 10:
                self.heatmaps.pop(0)

            self.calc_avg_heat(10)
            avg_heat = functions.apply_threshold(self.avg_heatmap, 10)

            # Find final boxes from heatmap using label function
            labels = label(avg_heat)
            if labels[1] > 0:
                print(labels[1], 'cars found')
                if draw:
                    plt.imshow(labels[0], cmap='gray')
                    plt.show()

                draw_img, car_centers = functions.draw_labeled_bboxes(np.copy(window_img), labels=labels,
                                                                      color=(1.0, 0, 1.0))

                if draw:
                    fig = plt.figure()
                    plt.subplot(121)
                    plt.imshow(draw_img)
                    plt.title('Car Positions')
                    plt.subplot(122)
                    # Visualize the heatmap when displaying
                    heatmap = np.clip(avg_heat, 0, 255)
                    plt.imshow(heatmap, cmap='hot')
                    plt.title('Heat Map')
                    fig.tight_layout()
                    plt.show()

                window_img = draw_img

        # Back to image space
        return window_img * 255

    def process_video(self, input, output):
        """
        Process video clip for vehicle detection
        :param input:
        :param output:
        :return:
        """
        clip1 = VideoFileClip(input)
        white_clip = clip1.fl_image(self.process_frame)
        white_clip.write_videofile(output, audio=False)

    def save(self):
        """
        Save a trained vehicle detector
        :return:
        """
        with open('detector_pickle.p', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            print('Saved detector to file')

    def calc_avg_heat(self, min):
        """
        This function calculates the sum of all heatmaps
        :return:
        """
        heat = 0.
        if len(self.heatmaps) == 0:
            self.avg_heatmap = heat
            # elif len(self.heatmaps) < min:
            # self.avg_heatmap = self.heatmaps.pop(-1)
        else:
            for i in range(0, len(self.heatmaps)):
                heat += self.heatmaps[i]
            self.avg_heatmap = heat
