This code runs a hand-engineered convolutional neural network (ConvNet). 
A hand-sign image dataset sourced from https://coursera.org/share/b5ada0e8a36a2bb04ed089d54f1ab25d was used. Some samples images of the training datasets are the following:

![Screenshot 2023-08-27 at 15 18 37](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/f6a928c2-6c71-4232-bff4-5e757182e02c)

This image is associated with a label number of 4. 

With a hand-engineered convolutional network implemented in the order of conv2D -> RELU -> max_pool (1st layer) -> conv2D (2nd layer) -> RELU -> max_pool -> flatten -> softmax (3rd layer), a training and testing accuracy of approximately 45% and 15% was achieved. The low accuracy can be attributed to the use of only a small portion of the entire dataset given the limitation of high computational cost for using for-loops in each layers. Next steps include the vectoriation of the computation and the addition of regularization algorithms to elevate the testing accuracy. 

Though the accuracy score was low, a plot of the cost vs iteration curve showed a consistent cost decline as shown below.  

![cost vs iteration conv](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/eae75ab0-07d5-45f5-9ffa-6d6ab05a9982)

Given the computational complexity of the operation, a tensorflow functional API was used to improve the accuracy scores. This code is available in a different file titled "Convolutional Neural Network API". 
