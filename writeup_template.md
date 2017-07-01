# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

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
* **1st convert the images to grayscale**
  * Converting into grayscale images are very important in this dataset. I tried 16 different CNN architecures without converting dataset into grayscale, the results are not satisfying. 
  * The main reason is that grayscale images simplifies the algorithm and reduces computational requirements. 
  * The grayscale images work best if there are limited amount of training data and illumination conditions were highly variable.
  * Check out the article: [Color-to-Grayscale: Does the Method Matter in Image Recognition?](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740), which gives great explanation why need color-to-grayscale.
  * I started a [discussion](https://discussions.udacity.com/t/validation-accuracy-88/293045/4) on Udacity Forum which might be helpful as well.
* **2nd normalize the image data**
  * image pixel values are from 0 to 255, which can form a large gap between largest value and smallest value. This gap impacts the deep neural network's learning capability significantly. Therefore, it is better to normalize the image data, scale it to low values. 
  * Normalization makes training less sensitive to the scale of features.
  * Normalization improves the convergence rate of gradient descent, make optimization well-conditioned.

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

**I started with color image data. I thought it is not necessary to change images into gray scale. As I knew famous VGG and AlexNet models are succeed on RGB images.**
* Started with LeNet. At EPOCH 10, Training Accuracy = 0.971, Validation Accuracy = 0.763. Clearly it is overfitting, so I added dropouts.
* LeNet + Dropout: At EPOCH 10, Training Accuracy = 0.868, Validation Accuracy = 0.722.  Dropout works. Let's try on more EPOCHS.
* More EPOCHES to train: At EPOCH 21,Training Accuracy = 0.973, Validation Accuracy = 0.805. Obviously it is overfitting again.
* Tried 16 different architecures and trained 16 times back and forth. Folowings are a few Accuracy Plottings.
![train1](/assets/training8.png)  ![train2](/assets/training9.png)  ![train3](/assets/training13.png) ![train4](/assets/training15.png)

* The common issue is overfitting. Lowest dropout I can try is 0.6. I tried 0.5, the model does not like it. It basically stopped learning.
* This [Google Doc](https://docs.google.com/document/d/1r1ZT1nIan5SOhfctdkrXNfES7yOc3hqESByBbwizVCo/edit#) has more detailed training process. Feel free to check it out. 

**After trying so many times, I was frustrated. I have to change the training strategy. I decided to work on dataset preprocessing.** 
* Convert image dataset to gray scale. This step is **GAME CHANGER**.
* Validation accuracy get to 94% just in 10 steps tranings.
* 20 EPOCHS:

![train5](/assets/gray_training1.PNG)  ![train6](/assets/gray_training2.PNG)

**General tactics for training the Deep Neural Networks:**
* start simple, make sure it is working.
  * with easy architecure, like LeNet.
  * small EPOCHS like 5,10.
  * appropriate learning rate, usually start with 0.001.
  * small batch_size, 32/64. This depends on the comoputer memory. Assuming that you start training on the local computer which may have less memory. 
* Make next move based on the results
  * if having big gap between training accuracy and validation accuracy, it means over-fitting. Need to apply DROPOUT. Dropout is usually added after Relu activation layer. It is not necessary to adapt dropout if the layer has MAX pooling. Max pooling function same purpose. 
  * Dropout's keep probability can start with 0.9. You can lower it based on the result. The suggested lowest value is 0.6.
  * If training accuracy is low, means the model is having hard time learning. Need to add more layers to make it deeper OR add more neurens to make it more wider OR BOTH. 
  * If tried many architecure and still struggling to improve, you may need to consider further preprocessing dataset.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found:
![alt text](/assets/new_imgs10.PNG)

**Pre-analyze**:
* image 2,3,8 are blur. The model may not recognize them well.
* image 10 has a backgrount that is familear with sign edge. It may cause problem.
* image 0,7 are pretty clear. The model should recognize them well.

#### 2. Predictions
![predictions](/assets/sofmax_10_top5.PNG)

**After-analyze**:
* The model prediction result: 9 correct, 1 incorrect. So the accuracy rate is 90%, which is lower than test accuray 95% of provided data.
* the model failed to recognize image 0 which is pretty clear. It predicted it as Speed limit (30km/h). It is supposed to b Speed limit (20km/h).
* I found that Speed limit (20km/h) has very small amount of training data, less than 250. This is main reason the model failed to predict it correctly. We have to enough data on this category to make sure the model learned to recognize it.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


