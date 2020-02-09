import tensorflow as tf
import cv2 as cv


model = tf.keras.applications.MobileNetV2()

img = cv.imread("index.jpeg")
output = model(img)