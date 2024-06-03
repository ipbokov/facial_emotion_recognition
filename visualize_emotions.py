import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("fer2013.csv", nrows=100)
pixels = data["pixels"].apply(lambda x: np.fromstring(x, sep=" ").reshape((48, 48)))
pixels = np.array(pixels.tolist()) / 255.0
emotion = np.array(data["emotion"])
class_names = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pixels[i, :], cmap="gray")
    plt.xlabel(class_names[emotion[i]])
plt.show()
