import numpy as np
import tensorflow as tf
import cv2 as cv

# left = "/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.1:1.0-video-index0"
#
# cam = cv.VideoCapture(left)
img = cv.imread('frame_00000.png')
cv.imshow('img', img)
# cv.waitKey(0)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/mobilenet.tflite")
print(interpreter)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

img = img.astype('uint8')
img = cv.resize(img, (300, 300))
print(img.shape)
input_image = np.asarray(img) #, dtype=np.float32)

input_image = np.expand_dims(input_image, 0)
print(input_image.shape)
# print(type(input_tensor))
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])
label = interpreter.get_tensor(output_details[1]['index'])

output_data = np.squeeze(output_data)
print(output_data)
scores = np.squeeze(scores)
label = np.squeeze(label)
print(label)

i = 0
# for box in output_data:
ymin = int(output_data[0][0]*300)
xmin = int(output_data[0][1]*300)
ymax = int(output_data[0][2]*300)
xmax = int(output_data[0][3]*300)
cv.rectangle(img, (xmin,ymin),(xmax,ymax), (0,0,255), thickness=4)
# img = np.squeeze(img, 0)
print(img.shape)
cv.imshow('img', img)
cv.waitKey(0)
# while True:
    # val, img = cam.read()
#     img = np.asarray(img, dtype=np.float32)
#     #img = img.astype('uint8')
#     img = cv.resize(img, (300,300))
#     input_tensor = tf.convert_to_tensor(img)
#     interpreter.set_tensor(input_details[0]['index'], input_tensor[tf.newaxis, ...])
#     interpreter.invoke()
#
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     scores = interpreter.get_tensor(output_details[2]['index'])
#     label = interpreter.get_tensor(output_details[1]['index'])
#
#     output_data = np.squeeze(output_data)
#     scores = np.squeeze(scores)
#     label = np.squeeze(label)
#     print(label)
#
#
#     i = 0
#     for box in output_data:
#         if scores[i] > 0.6:
#             ymin = int(box[0]*300)
#             xmin = int(box[1]*300)
#             ymax = int(box[2]*300)
#             xmax = int(box[3]*300)
#             cv.rectangle(img, (xmin,ymin),(xmax,ymax), (0,0,255))
#             try:
#             #if label[i] == 0:
#             #    print("person")
#             #elif label[i] == 64:
#             #    print("mouse")
#                 obj_label = labels[int(label[i])]
#                 print(str(obj_label) + ": " + str(scores[i]))
#                 cv.putText(img, str(obj_label), (xmin,ymin), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
#             except Exception as e:
#                 print("unable to find match")
#         i += 1
#     #cv.imshow("camera1", img)
#     if cv.waitKey(1) == 27:
#         break
# cam.release()
# cv.destroyAllWindows()
# #interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# #output_data = interpreter.get_tensor(output_details[0]['index'])
# #print(output_data)
