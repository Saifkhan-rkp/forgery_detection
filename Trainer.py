import os

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.legacy_tf_layers.core import Dropout
from tensorflow.python.keras.legacy_tf_layers.pooling import MaxPool2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.layers.convolutional import Conv2D


#-------------------------------------------------------------------------------------------------------------

class Config:
    CASIA1 = "../input/casia-dataset/CASIA1"
    CASIA2 = "../input/casia-dataset/CASIA2"
    autotune = tf.data.experimental.AUTOTUNE
    epochs = 30
    batch_size = 32
    lr = 1e-3
    name = 'xception'
    n_labels = 2
    image_size = (224, 224)
    decay = 1e-6
    momentum = 0.95
    nesterov = False

#-------------------------------------------------------------------------------------------------------------

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

#-------------------------------------------------------------------------------------------------------------

# def compute_ela_cv(path, quality):
#     temp_filename = 'temp_file_name.jpg'
#     SCALE = 15
#     orig_img = cv2.imread(path)
#     orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
#
#     cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
#
#     # read compressed image
#     compressed_img = cv2.imread(temp_filename)
#
#     # get absolute difference between img1 and img2 and multiply by scale
#     diff = SCALE * cv2.absdiff(orig_img, compressed_img)
#     return diff
#
# #-------------------------------------------------------------------------------------------------------------

image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 91).resize(image_size)).flatten() / 255.0


#-------------------------------------------------------------------------------------------------------------

X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real

#-------------------------------------------------------------------------------------------------------------

import random
path = '../input/casia-dataset/CASIA2/Au/'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')



#-------------------------------------------------------------------------------------------------------------

path = '../input/casia-dataset/CASIA2/Tp/'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

#-------------------------------------------------------------------------------------------------------------

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

import numpy as np
X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))


#-------------------------------------------------------------------------------------------------------------

def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    return model


model = build_model()
model.summary()


#-------------------------------------------------------------------------------------------------------------


from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
epochs = 24
batch_size = 32
init_lr = 1e-4
optimizer = Adam(lr = init_lr, decay = init_lr/epochs)


early_stopping = EarlyStopping(monitor = 'val_acc',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')



#-------------------------------------------------------------------------------------------------------------



from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
x_train2 = np.array(X_train, copy=True)
y_train2 = np.array(Y_train, copy=True)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(X_train)

print(type(X_train))

#earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')

validation_generator = datagen.flow(x_train2, y_train2, batch_size=32, subset='validation')
train_generator = datagen.flow(x_train2, y_train2,batch_size=32, subset='training')


# # fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_generator, epochs=epochs, validation_data = (X_val,Y_val), verbose = 1,callbacks = [early_stopping])


#-------------------------------------------------------------------------------------------------------------


model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


hist = model.fit(X_train,
                 Y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                validation_data = (X_val, Y_val),
                callbacks = [early_stopping])

#-------------------------------------------------------------------------------------------------------------

model.save('model_casia_run1.h5')