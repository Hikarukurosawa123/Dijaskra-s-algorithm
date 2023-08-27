This code runs a convolutional neural network (ConvNet) implemented using tensorflow functional API. 
A hand-sign image dataset sourced from https://coursera.org/share/b5ada0e8a36a2bb04ed089d54f1ab25d was used. 
Some samples images of the training datasets are the following:

![Screenshot 2023-08-27 at 15 18 37](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/ae86ce40-2cab-4865-bff9-947c67140c2f)

This image is associated with a label number of 4.

With a hand-engineered convolutional network implemented in the order of conv2D -> RELU -> max_pool (1st layer) -> conv2D (2nd layer) -> 
RELU -> max_pool -> flatten -> softmax (3rd layer), a training and testing accuracy of approximately 85% and 75% was achieved. 

Figures below show accuracy vs epoch graph and the loss vs epoch graph.  

![accuracy vs epoch](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/c04ceb35-2bc2-4413-a2df-5e9485cc0781)

![loss vs epoch ](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/e126be21-6ef5-4207-a03f-20d895889fed)

Next steps include the hyperparameter tuning to increase the testing score accuracy. 
