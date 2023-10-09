import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt  # Import matplotlib.pyplot as plt
import numpy as np
import os
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_dataset = train_datagen.flow_from_directory('C:/Users/hp/ml_internship/viewpoint/basedata/training',
                                                  target_size=(200, 200),
                                                  batch_size=3,
                                                  class_mode='binary')

validation_dataset = validation_datagen.flow_from_directory('C:/Users/hp/ml_internship/viewpoint/basedata/validation',
                                                            target_size=(200, 200),
                                                            batch_size=3,
                                                            class_mode='binary')

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
model_fit = model.fit(train_dataset,
                     steps_per_epoch=3,
                     epochs= 30,
                     validation_data=validation_dataset)
dir_path = 'C:/Users/hp/ml_internship/viewpoint/basedata/test'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '/' + i, target_size=(200, 200))  # Add parentheses for target_size
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print("you are happy")
    else:
        print("you are sad")
