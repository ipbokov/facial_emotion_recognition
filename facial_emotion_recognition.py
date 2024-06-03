import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

data = pd.read_csv('fer2013.csv')
pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotions = pd.get_dummies(data['emotion']).values

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train /= 255
X_val /= 255
X_test /= 255

model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 50
batch_size = 64

lr_scheduler = LearningRateScheduler(scheduler)

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler],
    shuffle=True
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Тестова точність моделі: {:.2f}%".format(test_acc * 100))

def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(history.history['accuracy'], label='Тренувальна точність')
    ax[0].plot(history.history['val_accuracy'], label='Валідаційна точність')
    ax[0].set_title('Точність моделі')
    ax[0].set_ylabel('Точність')
    ax[0].set_xlabel('Епоха')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Тренувальні втрати')
    ax[1].plot(history.history['val_loss'], label='Валідаційні втрати')
    ax[1].set_title('Втрати моделі')
    ax[1].set_ylabel('Втрати')
    ax[1].set_xlabel('Епоха')
    ax[1].legend()

    plt.show()

plot_training_history(history)

model.save('emotion_recognition_model.h5')
