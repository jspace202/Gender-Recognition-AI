import numpy as np
import cv2
import os
import glob
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

imgdims = (96,96,3)
data = []
labels = []

image_files = [f for f in glob.glob(r'C:\Users\jspac\OneDrive\Desktop\archive (1)1\Dataset\Validation\Male' + "/**/*", recursive=True) if not os.path.isdir(f)]

for img in image_files:

    image = cv2.imread(img)
    image = cv2.resize(image, (imgdims[0],imgdims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append(label)

data = np.array(data, dtype="float") / 255
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)
model = load_model('C:/Users/jspac/newgendermod.model')
score = model.evaluate(data, labels, verbose=0)
print('validation accuracy:', score[1])