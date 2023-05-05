import cv2
import os

face_cascade = cv2.CascadeClassifier('C:/Users/jspac/OneDrive/Desktop/Gender-Detection-master/Gender-Detection-master/haarcascade_frontalface_default.xml')

input_dir = 'C:/Users/jspac/OneDrive/Desktop/woman-image'
output_dir = 'C:/Users/jspac/OneDrive/Desktop/woman-crop'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    image = cv2.imread(os.path.join(input_dir, filename))
    
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    print(filename)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        output_filename = os.path.join(output_dir, 'face_' + filename)
        cv2.imwrite(output_filename, face)
