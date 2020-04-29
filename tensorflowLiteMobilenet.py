import numpy as np
import tensorflow as tf
import cv2 as cv
# left = "/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.1:1.0-video-index0"
#
# cam = cv.VideoCapture(left)
# img = cv.imread('/home/harry/git/mqp/AITraining/data/images/train/00000001.jpg')
# cv.imshow('img', img)
# # cv.waitKey(0)
# Load TFLite model and allocate tensors.
# with open('/home/harry/git/mqp/AITraining/models/tflite/labelmap.txt', 'r') as f:
#     label_names = f.read().split('\n')
#     label_names.remove('???')
#     print(label_names)




interpreter = tf.lite.Interpreter(model_path="/home/harry/git/mqp/AITraining/models/tflite/new_mv2.tflite")
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


cam = cv.VideoCapture("/home/harry/git/mqp/AITraining/Divers Fight the Invasive Lionfish   National Geographic.mp4")
print(cam.isOpened())
while True:
    ret, img = cam.read()
    # print(ret)
    if not ret:
        cam = cv.VideoCapture("/home/harry/git/mqp/AITraining/Divers Fight the Invasive Lionfish   National Geographic.mp4")

    img = cv.resize(img, (300, 300))
    # print(img.shape)
    input_image = np.asarray(img, dtype=np.float32)
    #
    input_image = np.expand_dims(input_image, 0)
    # input_image = (np.float32(input_image) - 127.5) / 127.5

    # print(input_image.shape)
    # print(type(input_tensor))
    # print(input_image.shape)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    #
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    labels = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
    indices = [index for index in range(0, 10)]
    print(labels)

    for index in indices:
        # print(scores[index], labels[index])
        box = boxes[index]
        ymin = int(box[0] * 300)
        xmin = int(box[1] * 300)
        ymax = int(box[2] * 300)
        xmax = int(box[3] * 300)
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=4)

    cv.imshow('img', img)
    cv.waitKey(1)
    #
