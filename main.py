import cv2
import functions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
from VehicleDetector import VehicleDetector
import pickle
import os
import pandas as pd
import sys
import shutil

print('Modules loaded.')

"""
Strategy to tackle

0. Calibrate the camera

1. for each frame of video

2. use sliding window technique to detect vehicles

3. wherever classifier returns a positive detection

4. record the position of the window in which the detection was made

5. in some case if detect same vehicle in overlapping windows or different scales
assign the position of the detection to the centroid of the overlapping windows

6. filter out false positives by determining which detections appear in one frame but not the next

7. once a high confidence detection, record how its centroid is moving from frame to frame
and estimate where it will appear in each subsquent frame

"""


def get_rect(img, xmin, ymin, xmax, ymax):
    h = (ymax - ymin)
    w = (xmax - xmin)
    box = img[ymin:ymin + h, xmin:xmin + w]
    # plt.imshow(box)
    # plt.show()
    return box


def read_object_set(path='./data/object-detection-crowdai/', file='labels.csv', dsize=(64, 64),
                    save_path='./data/vehicles/crowdai/', labels='labels.p'):
    if os.path.exists(save_path + labels):
        with open(save_path + labels, 'rb') as input:
            print('Loaded crowdai labels from file')
            return pickle.load(input)

    row = pd.read_csv(path + file)
    X = {}
    print('Reading {} entries'.format(row.size))

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    for i in range(0, row.size):
        if i % 1000 == 0:
            print('Entries processed: {}'.format(i))
        try:
            img = cv2.imread(path + row['Frame'][i])
            roi = get_rect(img, row['xmin'][i], row['ymin'][i], row['xmax'][i], row['ymax'][i])
            resized = cv2.resize(roi, dsize=dsize)
            # plt.imshow(resized)
            # plt.show()
            label = row['Label'][i]
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            mpimg.imsave(save_path + row['Frame'][i], resized)
            X.setdefault(label, []).append(save_path + row['Frame'][i])
        except:
            print('Error at image index {0}: {1}'.format(i, sys.exc_info()[0]))

    with open(save_path + labels, "wb") as file:
        pickle.dump(X, file)
        print('Imported crowdai and saved label file')

    print('Vehicles imported and saved as', save_path)
    return X


def read_vehicles():
    """
    Read in our vehicles and non-vehicles
    :return:
    """
    images = glob.glob('./data/*vehicles/*/*.png')
    cars = []
    notcars = []

    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        elif 'vehicles' in image:
            cars.append(image)
        else:
            print('Nothing')

    print('Cars: ', len(cars))
    print('Non Cars: ', len(notcars))
    return cars, notcars


def sample_hog(detect: VehicleDetector, cars, notcars):
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    image = image.astype(np.float32)
    image = image[:, :, 0]

    # Call our function with vis=True to see an image output
    features, hog_image = functions.get_hog_features(image, orient=detect.orient,
                                                     pix_per_cell=detect.pix_per_cell,
                                                     cell_per_block=detect.cell_per_block,
                                                     hog_vis=detect.hog_vis, hog_feature_vec=detect.hog_feature_vec)
    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image ')

    plt.subplot(122)
    plt.imshow(hog_image, cmap='hot')
    plt.title('HOG Visualization')
    fig.savefig('./output_images/example_hog.jpg')
    plt.show()


def explore_histogram(test_img, draw=True):
    """
    Show a color histogram
    :param test_img:
    :return:
    """
    functions.color_hist2(test_img, nbins=32, draw=draw)


def test_frame(detect):
    """
    Test detection on sample image
    :param detect:
    :return:
    """
    test_img = cv2.imread('./test_images/test6.jpg')
    window_img = detect.process_frame(test_img)
    plt.imshow(window_img)
    plt.show()


def run():
    # vehicles = read_object_set()

    always_new = False

    # check if we already trained a classifier and train if not yet exists
    if not always_new and os.path.exists('detector_pickle.p'):
        detect = load()
        print('Pretrained detector loaded.')
    else:
        print('Creating a new detector.')
        detect = VehicleDetector(color_space='YCrCb', orient=7, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                                 spatial_size=(32, 32), hist_bins=32, spatial_feat=True, hist_feat=True, hog_feat=True,
                                 overlap=0.9,
                                 hist_range=(0, 256), hog_vis=True, hog_feature_vec=True, C=1., ystart=390, ystop=660,
                                 scale=1.5)

        # -------------------------------------------------CALIBRATION--------------------------------------------------
        if not detect.calibrate_camera():
            return
        # -------------------------------------------------HISTOGRAM----------------------------------------------------
        # explore_histogram(test_img)
        # -------------------------------------------------READ_VEHICLES------------------------------------------------
        cars, notcars = read_vehicles()
        # cars += vehicles['Car']
        # -------------------------------------------------SAMPLE_HOG---------------------------------------------------
        # sample_hog(detect, cars, notcars)

        # -------------------------------------------------TRAIN--------------------------------------------------------
        n_predict = detect.hog_classify(cars, notcars, color_space=detect.color_space, pix_per_cell=detect.pix_per_cell,
                                        hog_channel=detect.hog_channel, cell_per_block=detect.cell_per_block,
                                        orient=detect.orient, sample_size=None, hist_bins=detect.hist_bins,
                                        hist_feat=detect.hist_feat, hog_feat=detect.hog_feat,
                                        spatial_feat=detect.spatial_feat, spatial_size=detect.spatial_size,
                                        hist_range=detect.hist_range)
        detect.save()

    # try first with test images
    # test_frame(detect)

    # available videos to test the detection
    input = ['./test_video', './project_video']
    index = 0
    detect.process_video(input[index] + '.mp4', input[index] + '_processed.mp4')


def load():
    """
    Load a trained classifier
    :return:
    """
    with open('detector_pickle.p', 'rb') as input:
        return pickle.load(input)


if __name__ == '__main__':
    run()
