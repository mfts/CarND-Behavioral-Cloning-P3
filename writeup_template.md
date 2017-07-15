# **Traffic Sign Recognition** 
---

## Setup
---

### Installation

Runs Jupyter Notebook in a Docker container with `udacity/carnd-term1-starter-kit` image from [Udacity][docker installation].

```
cd ~/src/CarND-Trafic-Sign-Classifier-Project
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```
Go to `localhost:8888`

**For training the model via GPUs**
To speed up training the model, I opted for the GPU-enabled AWS EC2 instance. Feel free to follow the Udacity AWS instructions [here][aws instructions].


## Reflection
---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator on Mac and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
or if you are using docker
```sh
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5 
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's model architecture for self-driving cars [here][nvidia model], which consists of 3 convolutions with 5x5 kernel size and 2 convolutions with 3x3 kernel size followed by 3 fully connected layers. (model.py lines 46-55)

The convolutions include RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 44) and a cropping layer to remove unnecessary noise from the images (model.py line 45). 

#### 2. Attempts to reduce overfitting in the model

I tried including a dropout layer but it didn't change much in the validation loss fluctuations.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 57).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded 2 laps staying inside the lanes (mostly center lane driving) and 1 additional lap with repeated recovering from left and right sides of the road back to the center of the lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA model for self-driving cars. Before that, I tested different models including LeNet and AlexNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


[docker installation]: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_docker.md
[aws instructions]: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/docker_for_aws.md
[nvidia model]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"