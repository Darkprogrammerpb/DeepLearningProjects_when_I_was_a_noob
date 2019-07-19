#!/usr/bin/env python
# coding: utf-8

# # 10 MONKEY SPECIES CLASSIFICATION USING TRANSFER LEARNING

# The data is obtained from the following Kaggle link:- <br>
# https://www.kaggle.com/slothkong/10-monkey-species <br>
# <div class="alert alert-block alert-info">
# <b>Aim:</b> This dataset is intended as a test case for fine-grain classification tasks, perhaps best used in combination with transfer learning.
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Overview of VGG16 Architecture:</b> A brief overview of VGG16 Architecture is given below in the diagram. Each box in the blocks (from Block 1 to Block 5) contains Convolution Number, Size tensor, Kernel size and stride in the order.

# <img src="architecture_vgg.jpg">

# # Step 1:- Importing libraries to be used for predictions
# * For the implementation of VGG16, we are going to use Keras library.
# * For any sort of visualization we are going to use matplotlib 

# In[1]:


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


# ## Step 1.1 Configuration sheet for parameter setting
# We will be using a configuration sheet that will enable us to set the parameters of the model to be used in the course of the program. A typical configuration sheet(present as configuration.xlsx) will look like below:-

#  <tr>
#     <td> <img src="config1.png" alt="Drawing" style="width: 650px;"/> </td>
#     <td> <img src="config2.png" alt="Drawing" style="width: 650px;"/> </td>
#     </tr>

# In[2]:


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


# # Step 2:- Reading from the text file
# * Here we are reading the text file and converting it into a pandas dataframe.
# * Also we are creating a class dicitonary in order to assign common names to the classified models to be used for prediciton 

# In[3]:


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


# # Step 3:- Displaying random images from training dataset 
# We will display random images of each monkey type along with their names 

# In[4]:


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


# # Step 4:-  Creating the model 

# In[5]:



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


# In[ ]:





# # Step 5:- Loading train and test image set
# We will load train and test images. But before that we will create a data generator for data augmentation

# In[6]:


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



# ##### Here we are loading the test and train images using the instange of ImageDataGenerator having specified values of its own parameters

# In[7]:


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


# ## Step 5.1 :- Setting up callback parameters to improve model fitting

# In[8]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,verbose=3,
                              patience=5, min_lr=0.001)
es        = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', baseline=None, 
                          restore_best_weights=True)


# # Step 6:- Fitting the model 
# <div class="alert alert-block alert-success">
# <b>Parameter interpretation:</b> We will be fitting the model with certain parameters and the meaning of its relevant parameters is explained in the diagram below 

# <img src="param1.jpg">

# In[9]:


model_history = model.fit_generator(
                                      train_generator,
                                      validation_data  = test_generator,
                                      epochs           = epochs,
                                      steps_per_epoch  = len(train_generator.filenames)//batch_sz,
                                      validation_steps = len(test_generator.filenames)//batch_sz,
                                      callbacks        = [reduce_lr,es]
                                    )


# # Step 7:- Performance measure of the model

# In[12]:


valid_steps           = len(test_generator.filenames)//batch_sz
valid_loss, valid_acc = model.evaluate_generator(test_generator, steps=valid_steps,verbose=True)


# In[15]:


print("Final validation accuracy: ",round(valid_acc*100,3))


# In[2]:


get_ipython().system('nbconvert -f blogger-html VGG16_Implementation.ipynb')


# In[ ]:




