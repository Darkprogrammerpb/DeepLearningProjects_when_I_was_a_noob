
# 10 MONKEY SPECIES CLASSIFICATION USING TRANSFER LEARNING

The data is obtained from the following Kaggle link:- <br>
https://www.kaggle.com/slothkong/10-monkey-species <br>
<div class="alert alert-block alert-info">
<b>Aim:</b> This dataset is intended as a test case for fine-grain classification tasks, perhaps best used in combination with transfer learning.
</div>


<div class="alert alert-block alert-success">
<b>Overview of VGG16 Architecture:</b> A brief overview of VGG16 Architecture is given below in the diagram. Each box in the blocks (from Block 1 to Block 5) contains Convolution Number, Size tensor, Kernel size and stride in the order.

<img src="architecture_vgg.jpg" style="width: 1000px;/">

# Step 1:- Importing libraries to be used for predictions
* For the implementation of VGG16, we are going to use Keras library.
* For any sort of visualization we are going to use matplotlib 


```python
### Step 1: Importing all possible keras library modules
import keras
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
### Importing all other necessary libraries 
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random 
```

    Using TensorFlow backend.
    

## Step 1.1 Configuration sheet for parameter setting
We will be using a configuration sheet that will enable us to set the parameters of the model to be used in the course of the program. A typical configuration sheet(present as configuration.xlsx) will look like below:-

 <tr>
    <td> <img src="config1.png" alt="Drawing" style="width: 650px;"/> </td>
    <td> <img src="config2.png" alt="Drawing" style="width: 650px;"/> </td>
    </tr>


```python
config_file_path    = os.getcwd() +r'\configuration.xlsx'
data_parameters     = pd.read_excel(config_file_path,sheet_name='Data_information')
model_parameters    = pd.read_excel(config_file_path,sheet_name='VGG16_PARAMETERS',index_col='Feature')
number_of_labels    = data_parameters['Label'].values[0]
number_of_columns   = data_parameters['Columns'].values[0]
batch_sz            = model_parameters.loc['batch_size'].values[0]
epochs              = model_parameters.loc['epochs'].values[0]
training_folder     = model_parameters.loc['training_folder'].values[0]
testing_folder      = model_parameters.loc['testing_folder'].values[0]
width               = model_parameters.loc['width'].values[0]
height              = model_parameters.loc['height'].values[0]
channels            = model_parameters.loc['channels'].values[0]
wghts_vgg16         = model_parameters.loc['weights'].values[0]
lyrs_to_train       = model_parameters.loc['layers_to_train'].values[0]
inc_top             = model_parameters.loc['include_top'].values[0]
pooling_vgg16       = model_parameters.loc['pooling'].values[0]
num_dense_layers    = model_parameters.loc['number_of_dense_layers'].values[0]
drpt_rate           = model_parameters.loc['Dropout_rate'].values[0]
hidden_unts         = model_parameters.loc['Hidden_units'].values[0]
loss_func           = model_parameters.loc['loss_function'].values[0]
training_path       = os.getcwd()+'\\'+training_folder
testing_path        = os.getcwd()+'\\'+testing_folder
```

# Step 2:- Reading from the text file
* Here we are reading the text file and converting it into a pandas dataframe.
* Also we are creating a class dicitonary in order to assign common names to the classified models to be used for prediciton 


```python
file_object      = open('monkey_labels.txt','r')
file_to_read     = file_object.readlines()
df               = pd.DataFrame(columns =[file_to_read[0].split(',')[i].strip() for i in range(number_of_columns)])
for i in range(1,number_of_labels+1):
    row          = [file_to_read[i].split(',')[j].strip() for j in range(number_of_columns)]
    df.loc[i]    = row
class_label_dict = dict(zip(df['Label'].values,df['Common Name'].values))
print('The class label dictionary is as follows ',class_label_dict)
print('The dataframe having information about different image class(attained from \'monkey_labels.txt\') is below:- ')
df
```

    The class label dictionary is as follows  {'n0': 'mantled_howler', 'n1': 'patas_monkey', 'n2': 'bald_uakari', 'n3': 'japanese_macaque', 'n4': 'pygmy_marmoset', 'n5': 'white_headed_capuchin', 'n6': 'silvery_marmoset', 'n7': 'common_squirrel_monkey', 'n8': 'black_headed_night_monkey', 'n9': 'nilgiri_langur'}
    The dataframe having information about different image class(attained from 'monkey_labels.txt') is below:- 
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Latin Name</th>
      <th>Common Name</th>
      <th>Train Images</th>
      <th>Validation Images</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>n0</td>
      <td>alouatta_palliata</td>
      <td>mantled_howler</td>
      <td>131</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>n1</td>
      <td>erythrocebus_patas</td>
      <td>patas_monkey</td>
      <td>139</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>n2</td>
      <td>cacajao_calvus</td>
      <td>bald_uakari</td>
      <td>137</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>n3</td>
      <td>macaca_fuscata</td>
      <td>japanese_macaque</td>
      <td>152</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>n4</td>
      <td>cebuella_pygmea</td>
      <td>pygmy_marmoset</td>
      <td>131</td>
      <td>26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>n5</td>
      <td>cebus_capucinus</td>
      <td>white_headed_capuchin</td>
      <td>141</td>
      <td>28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>n6</td>
      <td>mico_argentatus</td>
      <td>silvery_marmoset</td>
      <td>132</td>
      <td>26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>n7</td>
      <td>saimiri_sciureus</td>
      <td>common_squirrel_monkey</td>
      <td>142</td>
      <td>28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>n8</td>
      <td>aotus_nigriceps</td>
      <td>black_headed_night_monkey</td>
      <td>133</td>
      <td>27</td>
    </tr>
    <tr>
      <th>10</th>
      <td>n9</td>
      <td>trachypithecus_johnii</td>
      <td>nilgiri_langur</td>
      <td>132</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



# Step 3:- Displaying random images from training dataset 
We will display random images of each monkey type along with their names 


```python
for i in range(number_of_labels):
    key          = 'n'+str(i)
    monkey_type  = class_label_dict[key]
    path_image   = training_path+r'\n'+str(i)
    file_to_read = path_image+'\\'+random.choice(os.listdir(path_image))
    img          = mpimg.imread(file_to_read)
    plt.imshow(img,aspect = 'equal')
    plt.title(monkey_type)
    plt.tight_layout(True)
    plt.show()
```


![png](VGG16_Implementation_files/VGG16_Implementation_12_0.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_1.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_2.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_3.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_4.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_5.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_6.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_7.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_8.png)



![png](VGG16_Implementation_files/VGG16_Implementation_12_9.png)


# Step 4:-  Creating the model 


```python

nadam = optimizers.Nadam(lr = 0.00008,beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.000001)

model = Sequential()
vgg = VGG16(input_shape=[width,height,channels], weights=wghts_vgg16, include_top=eval(inc_top),pooling = pooling_vgg16)
for layer in vgg.layers[:-1*lyrs_to_train]:
    layer.trainable = False
for layers in vgg.layers:
    model.add(layers)
for i in range(num_dense_layers):
    model.add(Dense(units = hidden_unts,activation = 'relu'))
    model.add(Dropout(drpt_rate))
model.add(Dense(units=number_of_labels,activation='softmax'))
model.compile(
  loss=loss_func,
  optimizer=nadam,
  metrics=['accuracy']
)

```

    WARNING:tensorflow:From C:\Users\Batfleck\Anaconda3\envs\keras-gpu\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From C:\Users\Batfleck\Anaconda3\envs\keras-gpu\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python

```

# Step 5:- Loading train and test image set
We will load train and test images. But before that we will create a data generator for data augmentation


```python
image_data_params  = pd.read_excel(config_file_path,sheet_name='image_data_preprocessing',index_col='Features')

image_data_gen  = ImageDataGenerator(
                                      rescale                 = 1./255,
                                      rotation_range          = image_data_params.loc['rotation_range'].values[0],
                                      width_shift_range       = image_data_params.loc['width_shift_range'].values[0],
                                      height_shift_range      = image_data_params.loc['height_shift_range'].values[0],
                                      shear_range             = image_data_params.loc['shear_range'].values[0],
                                      zoom_range              = image_data_params.loc['zoom_range'].values[0],
                                      horizontal_flip         = eval(image_data_params.loc['horizontal_flip'].values[0]),
                                      fill_mode               = 'nearest',
                                      preprocessing_function  = preprocess_input
                                    )



```

##### Here we are loading the test and train images using the instange of ImageDataGenerator having specified values of its own parameters


```python
train_generator = image_data_gen.flow_from_directory(
                                                      training_path,
                                                      target_size = [width,height],
                                                      shuffle     = True,
                                                      batch_size  = batch_sz,
                                                    )
test_generator  = image_data_gen.flow_from_directory(
                                                      testing_path,
                                                      target_size = [width,height],
                                                      shuffle     = True,
                                                      batch_size  = batch_sz,
                                                    )


```

    Found 1098 images belonging to 10 classes.
    Found 272 images belonging to 10 classes.
    

## Step 5.1 :- Setting up callback parameters to improve model fitting


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,verbose=3,
                              patience=5, min_lr=0.001)
es        = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', baseline=None, 
                          restore_best_weights=True)
```

# Step 6:- Fitting the model 
<div class="alert alert-block alert-success">
<b>Parameter interpretation:</b> We will be fitting the model with certain parameters and the meaning of its relevant parameters is explained in the diagram below 

<img src="param1.jpg" style="width: 800px;"/>


```python
model_history = model.fit_generator(
                                      train_generator,
                                      validation_data  = test_generator,
                                      epochs           = epochs,
                                      steps_per_epoch  = len(train_generator.filenames)//batch_sz,
                                      validation_steps = len(test_generator.filenames)//batch_sz,
                                      callbacks        = [reduce_lr,es]
                                    )


```

    WARNING:tensorflow:From C:\Users\Batfleck\Anaconda3\envs\keras-gpu\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From C:\Users\Batfleck\Anaconda3\envs\keras-gpu\lib\site-packages\tensorflow\python\ops\math_grad.py:102: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Deprecated in favor of operator or tf.math.divide.
    Epoch 1/20
    34/34 [==============================] - 48s 1s/step - loss: 2.0112 - acc: 0.2886 - val_loss: 1.1967 - val_acc: 0.6406
    Epoch 2/20
    34/34 [==============================] - 43s 1s/step - loss: 0.9298 - acc: 0.6837 - val_loss: 0.5977 - val_acc: 0.8208
    Epoch 3/20
    34/34 [==============================] - 39s 1s/step - loss: 0.4781 - acc: 0.8379 - val_loss: 0.4421 - val_acc: 0.8583
    Epoch 4/20
    34/34 [==============================] - 40s 1s/step - loss: 0.4018 - acc: 0.8743 - val_loss: 0.3023 - val_acc: 0.8958
    Epoch 5/20
    34/34 [==============================] - 41s 1s/step - loss: 0.2148 - acc: 0.9384 - val_loss: 0.2455 - val_acc: 0.9292
    Epoch 6/20
    34/34 [==============================] - 41s 1s/step - loss: 0.2043 - acc: 0.9373 - val_loss: 0.2559 - val_acc: 0.9250
    Epoch 7/20
    34/34 [==============================] - 41s 1s/step - loss: 0.1644 - acc: 0.9430 - val_loss: 0.1926 - val_acc: 0.9458
    Epoch 8/20
    34/34 [==============================] - 40s 1s/step - loss: 0.1437 - acc: 0.9574 - val_loss: 0.2437 - val_acc: 0.9250
    Epoch 9/20
    34/34 [==============================] - 42s 1s/step - loss: 0.0972 - acc: 0.9678 - val_loss: 0.2062 - val_acc: 0.9250
    Epoch 10/20
    34/34 [==============================] - 41s 1s/step - loss: 0.1166 - acc: 0.9594 - val_loss: 0.1944 - val_acc: 0.9297
    Epoch 11/20
    34/34 [==============================] - 43s 1s/step - loss: 0.1062 - acc: 0.9668 - val_loss: 0.2127 - val_acc: 0.9167
    Epoch 12/20
    34/34 [==============================] - 43s 1s/step - loss: 0.0839 - acc: 0.9686 - val_loss: 0.2305 - val_acc: 0.9333
    Epoch 13/20
    34/34 [==============================] - 45s 1s/step - loss: 0.0605 - acc: 0.9816 - val_loss: 0.1904 - val_acc: 0.9333
    Epoch 14/20
    34/34 [==============================] - 44s 1s/step - loss: 0.0622 - acc: 0.9825 - val_loss: 0.2351 - val_acc: 0.9292
    Epoch 15/20
    34/34 [==============================] - 44s 1s/step - loss: 0.0518 - acc: 0.9834 - val_loss: 0.1900 - val_acc: 0.9500
    Epoch 16/20
    34/34 [==============================] - 45s 1s/step - loss: 0.0520 - acc: 0.9834 - val_loss: 0.2511 - val_acc: 0.9125
    Epoch 17/20
    34/34 [==============================] - 43s 1s/step - loss: 0.0346 - acc: 0.9936 - val_loss: 0.2158 - val_acc: 0.9375
    Epoch 18/20
    34/34 [==============================] - 45s 1s/step - loss: 0.0247 - acc: 0.9899 - val_loss: 0.2986 - val_acc: 0.9208
    Epoch 19/20
    34/34 [==============================] - 44s 1s/step - loss: 0.0283 - acc: 0.9888 - val_loss: 0.2382 - val_acc: 0.9375
    Epoch 20/20
    34/34 [==============================] - 44s 1s/step - loss: 0.3004 - acc: 0.9292 - val_loss: 0.3135 - val_acc: 0.9208
    

# Step 7:- Performance measure of the model


```python
valid_steps           = len(test_generator.filenames)//batch_sz
valid_loss, valid_acc = model.evaluate_generator(test_generator, steps=valid_steps,verbose=True)

```

    8/8 [==============================] - 7s 936ms/step
    


```python
print("Final validation accuracy: ",round(valid_acc*100,3))
```

    Final validation accuracy:  94.167
    


```python

```


```python

```
