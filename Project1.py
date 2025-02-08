# This project is for image classification using CNN
#It classifies images into 10 different categories ('airplane','automobiles','bird','cat','deer','dog','frog','house','ship','truck')

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets,layers,models
from keras import preprocessing
from keras_preprocessing import image
import numpy as np

#Downloading dataset
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()

#Normalizing pixel values between 0 & 1
train_images=train_images/255.0
test_images=test_images/255.0

#10 classes names/

class_name=['airplane','automobiles','bird','cat','deer','dog','frog','house','ship','truck']

    
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3))) #it will find features from image
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu')) #this layer checks more detailed features
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10)) # as i have 10 classes

#model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=10,
                  validation_data=(test_images,test_labels))
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalizing the image
    return img_array

def classify_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Predicting First image 
img_path = r'C:\Users\anmol\OneDrive\Pictures\R.jpeg' # Path of my image
img_array = load_and_preprocess_image(img_path)
predicted_class = classify_image(model, img_array)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'house', 'ship', 'truck']
predicted_class_name = class_names[predicted_class]
print(f'The predicted class is: {predicted_class_name}')


# Predicting Second image
img_path2 = r'C:\Users\anmol\OneDrive\Pictures\Car.jpeg' # Path of image
img_array2 = load_and_preprocess_image(img_path2)
predicted_class= classify_image(model, img_array2)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'house', 'ship', 'truck']
predicted_class_name2 = class_names[predicted_class]

print(f'The predicted class is: {predicted_class_name2}')
