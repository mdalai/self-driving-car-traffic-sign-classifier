#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mdalai/self-driving-car-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12360.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![summary plot](/assets/bar_plot.PNG)

### Design and Test a Model Architecture

#### 1. Preprocessed the image data.
* Convert the images to grayscale
* normalize the image data


#### 2. The model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x32 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x32 |
| RELU					|												|
| Dropout	      	| 0.6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x32 |
| RELU					|												|
| Dropout      	| 0.6	|
| Flatten      	| 8x8x32 = 2048	|
| Fully connected		| (2048, 128)				|
| Dropout      	| 0.6	|
| Fully connected		| (128, 84)				|
| Dropout      	| 0.6	|
| Fully connected		| (84, 43)				|
|						|												|
 


#### 3. Train the model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
* optimizer: AdamOptimizer
* batch size: 128
* number of epochs: 20
* learning rate: 0.001

#### 4. Training process.
My final model results were:
* training set accuracy of 99%.
* validation set accuracy of 96.6%.
* test set accuracy of 95.1%.

I started with color image data. I thought it is not necessary to change images into gray scale. As I knew famous VGG and AlexNet models are succeed on RGB images.
* Started with LeNet. At EPOCH 10, Training Accuracy = 0.971, Validation Accuracy = 0.763. Clearly it is overfitting, so I added dropouts.
* LeNet + Dropout: At EPOCH 10, Training Accuracy = 0.868, Validation Accuracy = 0.722.  Dropout works. Let's try on more EPOCHS.
* More EPOCHES to train: At EPOCH 21,Training Accuracy = 0.973, Validation Accuracy = 0.805. Obviously it is overfitting again.
* Tried 16 different architecures and trained 16 times back and forth. Folowings are a few Accuracy Plottings.
![train1](/assets/training8.png)  ![train2](/assets/training9.png)  ![train3](/assets/training13.png) ![train4](/assets/training15.png)

* The common issue is overfitting. Lowest dropout I can try is 0.6. I tried 0.5, the model does not like it. It basically stopped learning.
* This [Google Doc](https://docs.google.com/document/d/1r1ZT1nIan5SOhfctdkrXNfES7yOc3hqESByBbwizVCo/edit#) has more detailed training process. Feel free to check it out. 

After trying so many times, I was frustrated. I have to change the training strategy. I decided to work on dataset preprocessing. 
* Convert image dataset to gray scale. This step is **GAME CHANGER**.
* Validation accuracy get to 94% just in 10 steps tranings.
* 20 EPOCHS:

![train5](/assets/gray_training1.PNG)  ![train6](/assets/gray_training2.PNG)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found:

![alt text](/test/test1.png) ![alt text](/test/test2.png) ![alt text](/test/test3.png) ![alt text](/test/test4.png) ![alt text](/test/test5.png)

#### 2. Predictions
![predictions](/assets/sofmax_top5.PNG)


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


