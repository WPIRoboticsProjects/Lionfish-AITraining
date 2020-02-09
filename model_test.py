import tensorflow as tf
import numpy as np
import cv2 as cv
import datetime
image = cv.imread("bear.jpg")

image = np.asarray(image)

mobile = tf.saved_model.load("models/ssd_mobilenet_v2/saved_model")
mobile = mobile.signatures['serving_default']
inception = tf.saved_model.load("models/ssd_inception_v2/saved_model")
inception = inception.signatures['serving_default']
resnet = tf.saved_model.load("models/ssd_resnet50_v1/saved_model")
print(resnet)
resnet = resnet.signatures['serving_default']
print(resnet)



input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]
mobile_output = mobile(input_tensor)
inception_output = inception(input_tensor)
resnet_output = resnet(input_tensor)
print(resnet_output)
