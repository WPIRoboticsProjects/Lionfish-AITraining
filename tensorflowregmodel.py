import numpy as np
import tensorflow as tf
import cv2 as cv

loaded = tf.saved_model.load("/home/harry/git/mqp/AITraining/models/mobilenet_output_inference_graph/saved_model")
print(list(loaded.signatures.keys()))  # ["serving_default"]
infer = loaded.signatures["serving_default"]

# img = cv.resize(img, (300, 300))
# print(img.shape)
# input_image = np.asarray(img, dtype=np.float32)
#
# input_image = np.expand_dims(input_image, 0)
# print(input_image.shape)
# # print(type(input_tensor))
# interpreter.set_tensor(input_details[0]['index'], input_image)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# scores = interpreter.get_tensor(output_details[2]['index'])
# label = interpreter.get_tensor(output_details[1]['index'])

# output_data = np.squeeze(output_data)
# print(output_data)
# scores = np.squeeze(scores)
# label = np.squeeze(label)
# print(label)

# i = 0
# # for box in output_data:
# ymin = int(output_data[0][0]*300)
# xmin = int(output_data[0][1]*300)
# ymax = int(output_data[0][2]*300)
# xmax = int(output_data[0][3]*300)
# cv.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 0, 255), thickness=4)
# # img = np.squeeze(img, 0)
# print(img.shape)
# cv.imshow('img', img)
# cv.waitKey(0)

cam = cv.VideoCapture("Divers Fight the Invasive Lionfish   National Geographic.mp4")
while True:
    ret, img = cam.read()

    if not ret:
        cam = cv.VideoCapture("Divers Fight the Invasive Lionfish   National Geographic.mp4")

    img = cv.resize(img, (300, 300))
    # print(img.shape)
    input_image = np.asarray(img, dtype=np.float32)

    input_image = np.expand_dims(input_image, 0)
    # print(input_image.shape)
    # print(type(input_tensor))
    # output = infer(tf.convert_to_tensor(input_image, dtype=tf.uint8))
    # print(output)
    boxes = np.squeeze(output['detection_boxes'])
    print(np.squeeze(output['detection_scores']))
    print(np.squeeze(output['detection_classes']))
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # output_data = np.squeeze(output_data)
    # scores = interpreter.get_tensor(output_details[2]['index'])
    # label = interpreter.get_tensor(output_details[1]['index'])
    # print(np.squeeze(label))
    for i in range(0, len(boxes)):
        ymin = int(boxes[i][0] * 300)
        xmin = int(boxes[i][1] * 300)
        ymax = int(boxes[i][2] * 300)
        xmax = int(boxes[i][3] * 300)
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=4)
    #
    cv.imshow('img', img)
    cv.waitKey(1)