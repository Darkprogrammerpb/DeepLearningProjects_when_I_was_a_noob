# VGG16 Architecture 
A typical VGG16 architecture looks like this.
Each box in the Convolution and Pooling Block from Blocks 1 to 5, contains the Convolution Number, Image tensor(representing the height, width and features), kernal size and stride.

![architecture_vgg](https://user-images.githubusercontent.com/51089715/61504390-c98dae80-a9f8-11e9-8596-f38e73b4cb67.jpg)

# A brief overview of Training Data, Batch, Epochs and Batch size
![param1](https://user-images.githubusercontent.com/51089715/61504563-749e6800-a9f9-11e9-816d-3c88bbf63130.jpg)

In the code, I have used a configuration sheet which contains all the parameters needed as an input for both the model as well as for data augmentation. A snippet of the configuration file (saved as configuration.xlsx)is shown below:- 
![config2](https://user-images.githubusercontent.com/51089715/61504631-b9c29a00-a9f9-11e9-9a01-d243f812e36f.png)
![config1](https://user-images.githubusercontent.com/51089715/61504632-b9c29a00-a9f9-11e9-93c9-b63e7c3d4492.png)
