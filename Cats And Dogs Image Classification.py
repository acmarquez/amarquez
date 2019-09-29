# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:03:51 2019

@author: Allen
"""
'''
Cats and Dogs CNN
'''
import os, shutil
#Original path where we unzipped
original_dataset_dir = 'C:\\Users\\Allen\\Desktop\\School\\Math 385\\Datasets\\dogs vs cats'
# declaring a new directory we want to create
base_dir = 'C:\\Users\\Allen\\Desktop\\School\\Math 385\\Datasets\\dogs vs cats\\smaller_data'
os.mkdir(base_dir)
'''
===============================================================================
                Directories for the training, validation, and test splits:
    1)Training: data used to initially fit our model;used to fit the parameters (e.g. weights of connections between neurons in artificial neural networks) of the model.
   2) Validation: The validation dataset provides an unbiased evaluation of a model fit on the training dataset while tuning the model's hyperparameters. Validation datasets can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset.
    3)Testing: dataset used to provide an unbiased evaluation of a final model fit on the training dataset.
===============================================================================
'''
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# directory with cat training pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# directory with dog training pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# directory with cat validation pcitures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# directory with dog validation pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# directory with cat test set pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# directory with dog test set pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
'''
===============================================================================
Copying our original data to their respective directories
===============================================================================
'''
# Copies the first 1000 cat images to train_cats dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 cat images to validation_cats dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 cat images to test_cats dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the first 1000 dog images to train_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to validation_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to test_cats dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
'''
===============================================================================
Lets quickly check how many pictures are in each folder,
the way we split the data gives us a balanced binary classification problem
===============================================================================
'''
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
'''
===============================================================================
Building our network

Let A be and mxn matrix

convolving - dot product of filter with 3x3 block from input, then stores it in A1,1,...A1,n(ect),
then we are left with a new representation of the stored dot products, and will matrix of dot prodcuts will be the output of this layer and so on and so on.

Max pooling is typically added after a conv layer, it reduces the dimensionality of images,by reducing the 
number of pixels in the output from the output of previous conv layer. Define an nxn region as the corresponding filter for the max pooling operation, then we define a stride, meaning how many pixels do we want to move as it slide across the image, then we go to our conv output, then get the max value of the first nxn block and store it the full output of the max pool output, then move by our stride, then do max operation and repeat, the move down by our stride, and repeat.Lastly giving us a new representation of the image 

Look up flatten....

Because were attacking a binary classification problem, well end up with a single unit (dense layer of size 1) and a sigmoid activation. This unit will endcode the probablity that the network is looking at one class of the other.
===============================================================================
'''
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
===============================================================================
For the compilation step, well go with RMSprop optimizer, because the network ended with a single sigmoid unit, and well use binary cross entropy as the loss.
===============================================================================
'''
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
'''
===============================================================================
5.2.4 Data preproccesing

Data should be formatted into appropriately preprocessed floating point tensors before being fed into the network.

1. Read the picture files.
2. Decode the JPEG content to RGB grids of pixels
3. Convert these into floating point tensors
4. Rescale the pixel values (between 0 and 255) to the [0,1] interval, since nueral networks prefer to deal with small input values.
===============================================================================
'''
from keras.preprocessing.image import ImageDataGenerator
#rescale images by 1/255
train_datagen = ImageDataGenerator(rescale= 1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        #target directory
        train_dir,
        target_size=(150, 150),
        batch_size = 20,
        class_mode= 'binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
 print('data batch shape:', data_batch.shape)
 print('labels batch shape:', labels_batch.shape)
 break
'''
===============================================================================
Lets fit the model using the generator
steps_per_epoch: how many samokes to draw from before declaing and epoch over

Fitting the model using a batch generator
===============================================================================
'''

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
#save the model
model.save('cats_and_dogs_small_1.h5')
'''
===============================================================================
5.10 Displaying curves of loss and accuracy during training
===============================================================================
'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
'''
===============================================================================
We already now about techniques hat can help mitigate overfitting, such as drop out and weight decay(L2 Regularization)

We’re now going to work with a new one, specific to computer vision and used almost universally
when processing images with deep-learning models: data augmentation.
5.2.5 Using data augmentation

Data augmentation takes the approach of generating more training data
from existing training samples, by augmenting the samples via a number of random
transformations that yield believable-looking images.

-rotation_range is a value in degrees (0–180), a range within which to randomly
rotate pictures.
- width_shift and height_shift are ranges (as a fraction of total width or
height) within which to randomly translate pictures vertically or horizontally.
- shear_range is for randomly applying shearing transformations.
- zoom_range is for randomly zooming inside pictures.
- horizontal_flip is for randomly flipping half the images horizontally—relevant
when there are no assumptions of horizontal asymmetry (for example,
real-world pictures).
- fill_mode is the strategy used for filling in newly created pixels, which can
appear after a rotation or a width/height shift.
===============================================================================
'''
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''
===============================================================================
5.12 Displaying some randomly augmented training images
===============================================================================
'''
# Module with imagepreprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]
# choose single image to augment
img_path = fnames[3]
# reads the image and resizes it
img = image.load_img(img_path, target_size=(150, 150))
#Converts it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)
#Reshapes it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
'''
Generates batches of randomly transformed images. Loops indefinitely, so you need to break the
loop at some point!
'''
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
'''
===============================================================================
But the inputs it sees are still heavily intercorrelated,
because they come from a small number of original images—you can’t produce
new information, you can only remix existing information. As such, this may not
be enough to completely get rid of overfitting. To further fight overfitting, you’ll also
add a Dropout layer to your model, right before the densely connected classifier.
===============================================================================
'''
# 5.13 Defining a new convnet that includes dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])
'''
===============================================================================
5.14 Training the convnet using data-augmentation generators
===============================================================================
'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
#Note that the validation data shouldn’t be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)
#save the model
model.save('cats_and_dogs_small_2.h5')
'''
===============================================================================

===============================================================================
'''













