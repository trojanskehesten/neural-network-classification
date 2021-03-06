# Neural Network Classification
Classification objects on the aerial photos using neural networks  
Scheme of neural network please look at the file ["Description.png"](https://github.com/trojanskehesten/neural-network-classification/blob/master/Description.png).  
  
There are 2 models:  
  1. softmax model with 1 layer (["Easy_NN.py"](https://github.com/trojanskehesten/neural-network-classification/blob/master/Easy_NN.py))  
  2. convolutional neural network with 9 layers (["Convolutional_NN.py"](https://github.com/trojanskehesten/neural-network-classification/blob/master/Convolutional_NN.py)).
The file ["Prepare_data.py"](https://github.com/trojanskehesten/neural-network-classification/blob/master/Prepare_data.py) is used for preparing dataset for neural network.
  
Dataset for training was created by me using aerial orthophoto from unmanned aerial vehicle of Mavinci GmbH (ftp://mavinci.de). It is consist of 3384 aerial images of 3 objects (trees, grassland and roads).
The dataset is possible to download [here](https://yadi.sk/d/7Q5tMVM83YwSw7).  
  
The training is very hard for calculation. For the task please use powerfull computer or [google cloud](https://cloud.google.com/).
![DescriptionOfNeuralNetwork](https://github.com/trojanskehesten/neural-network-classification/blob/master/Description.png)
