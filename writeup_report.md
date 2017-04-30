# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[removed_95pct_straight_drive]: ./writeup/removed_95pct_straight_drive.png
[straight_drive_spike]: ./writeup/straight_drive_spike.png
[data_distribution_track_1]: ./writeup/data_distribution_track_1.png
[data_distribution_track_1_2]: ./writeup/data_distribution_track_1_2.png
[data_distribution_all_tracks]: ./writeup/data_distribution_all_tracks.png
[image_crop_after]: ./writeup/image_crop_after.png
[image_crop_before]: ./writeup/image_crop_before.png
[left_center_right]: ./writeup/left_center_right.png
[left_right_flip]: ./writeup/left_right_flip.png
[last_model_loss_chart]: ./writeup/last_model_loss_chart.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* data.py containing the functions to read the data
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python3 drive.py --preprocess-yuv model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is reimplementation of the [NVIDIA architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) consists of a convolution neural network with multiple 5x5 and 3x3 filters and depths between 24 and 64 (`model.py`, function `create_model_nvidia`)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, dropout layers can be enabled by passing the `dropout` configuration parameter (`--model-dropout` command line flag). The dropout is enabled only in the full-connected layers only. Additionally L2 regularization can be enabled by providing the `l2_regularizer` configuration parameter to the model (`--model-l2-regularizer` command line flag).

The dropout for the inputs of the first FC layer was set to be much lower than the dropout in hidden layers. The model was trained with various values for the dropout, but the final model does not have drop out enabled, as it was not allowing it to pass the last tough turn on track 2 (jungle).

The early stopping was the main method to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The code for loading the data is implemented in the function `preload_data_groupped` in `data.py`. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an Adam optimizer (in `train_model` function), so the learning rate was not tuned manually, but can be provided as a command line flag.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the sides of the road, I also heavily employed the data from the second track (jungles).

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try the LeNet model - that did not drive well, then the choice was to use the NVIDIA model, that was already tested and verified in real world, where it was controlling just a steering angle using image from the camera, the only difference should be real environment versus simulator.

The loss function for the model was `mse` (mean square error).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Using the command line flag `--validsize` of the `model.py`, the split proportion can be controlled. First training quickly discovered that model was overfitting, as error on the training set was going down, but error on a validation set was going up.

To combat the overfitting, I added the following:

1. Dropouts into the FC layer.
2. Early stopping, that monitors validation error value and if it does not improve for configured number of epochs (`--early-stopping-patience`), the training stops. The last last improvement is stored.
3. Gathered more data.

The final step was to run the simulator to see how well the car was driving around track one. It did pretty well, so I tested on the track 2 (jungle), and it was failing on a first turn. To improve the behavior, I changed the crop window in such a way that car is more focused on a road (by moving it lower). Also I gathered additional data around the cases where the car was failing.

Before:

![image_crop_before]

After:

![image_crop_after]

I also tried to use transfer learning to use the VGG16 as a feature extractor, and train it on streering data, but the model was so huge and slow to train, that I decided to not pursue that path.

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

The model was able to drive over the track 2 (jungle), and about 5 minutes into the track 2 (mountains), which it has never seen before, which shows that the resulting model generalized quite well.


#### 2. Final Model Architecture

The final model architecture (function `create_model_nvidia_4` in `model.py`) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Size | Strides | Activation |
| --- | --- | --- | --- |
| Cropping | | | |
| Normalization | | |
| Convolutional | 24x5x5 | 2x2 | RELU |
| Convolutional | 36x5x5 | 2x2 | RELU |
| Convolutional | 48x5x5 | 2x2 | RELU |
| Convolutional | 64x3x3 |  | RELU |
| Convolutional | 64x3x3 |  | RELU |
| Convolutional | 64x3x3 |  | RELU |
| Fully connected | 100 |  | RELU |
| Fully connected | 50 |  | RELU |
| Fully connected | 10 |  | RELU |
| Fully connected | 1 |  | RELU |

You may find it useful to see the visualization in Figure 4 in ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![left_center_right]

The visualization of the steering angles distribution showed that the dataset is not balanced and there is a huge spike of straight driving. The code was implemented to drop configured percent of the records with steering angle 0. That could lead the model to be biased driving straight. Here is the distribution of angles before and after applying the filter:

| Before filter | After filter |
| --- | --- |
| ![straight_drive_spike] | ![removed_95pct_straight_drive] |

Then I used recorded the same laps using joystick instead of the keyboard to obtain more smooth steering angles:

| Lap 1 keyboard | Lap 1 joystick |
| --- | --- |
| ![removed_95pct_straight_drive] | ![data_distribution_track_1] |

The problem here is that there are more left turns than right turns. To fix that, the data generator (`generate_data`) that is used when model is trained, flips images over vertical axis and changes the sign of the steering angle with 50% probability. As the model is not obtaining any information from anywhere else besides the lane itself (i.e. no risk of flipping road signs), this should make the model not biased to a particular turn direction.

![left_right_flip]

I then recorded the car driving on the center of the track 2, trying to pass it as well. That gave a lot of additional steering angles, as it bended more.

To simulate recovering from the left and right of the road sides, the data is augmented in the data generator on the fly: it uses images from the left and right cameras and adjusts steering angle to a configured value. The idea behind it is that the right camera sees the right lane marking closer to it, so the steering angle should be adjusted more to the left, than it is for the center camera. Similarly for the left camera. I assumed it is difficult to calculate the desired adjustment using geometry, as that would required to know what is the distance between the cameras, what is the desired recovery time, and potentially, what is the current car speed. So the adjustment of 0.2 rad was obtained experimentally so that a car recovers and does not oscillate around lane center a lot.

The steering angles distribution (excluding augmentation performed in the training generator):

![data_distribution_track_1_2]

This data set contained 6053 data points.

Additional data was gathered to cover tough cases for the second track, resulting in the final data set:

![data_distribution_all_tracks]

The final data set contained 9368 data points. After augmentation in the generator, this should result in `6 * 9368 = 56208` records in total (images from 3 cameras and flipped and non-flipped versions).

The data is preprocessed, converting the RGB color space into YUV. There were two reasons behind that: first that RGB is not the best color space for images recognition (e.g. car classifier worked better with YUV, but from other hand, CNN is more suitable to capture features from images than SVM), the second reason was that YUV was used in the original paper from the NVIDIA.

The data is grouped in the different directories, so that it is possible to select on which subsets to train on (e.g. only on track 1, or track 2, on both, include or exclude additional tough cases from track 2, etc.)

When data is loaded (in the function `preload_data_groupped` in `data.py`), every subset of data is preprocessed split into the training and test sets separately. The reason for that is to ensure that the training set and validation set contains equal portions of the data from different subsets, otherwise what was observed, was that the model would be trained mostly on the data from one track, and then fail to validate on the data that would have only track 2. Also I wanted to ensure that touch cases would be also included into the training set. The data was shuffled before splitting into the training and validation sets. Then

The model was trained for at most 25 epochs, the early stopping allowed to select the optimal model and avoid overfitting. The patience of 3 epochs was used. The final model required 16 epochs.

![last_model_loss_chart]