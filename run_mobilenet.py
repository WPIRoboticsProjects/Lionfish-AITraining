import numpy as np
import os
# from utils import label_map_util
# from utils import visualization_utils as vis_util
import tensorflow as tf
import cv2 as cv

left = "/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.1:1.0-video-index0"

# Define the video stream
cap = cv.VideoCapture(left)  # Change only if you have more than one webcams


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_MODEL = 'mobilenet/optimized_model.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mobilenet/data-inception-lionfish_lionfish_label_map.pbtxt'

# Number of classes to detect
NUM_CLASSES = 3

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# # Loading label map
# # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)


# # Helper code
# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape(
#         (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            ret, img = cap.read()

            img = cv.resize(img, (300, 300))
            image_np = np.asarray(img).astype('uint8')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8)

            # Display output
            cv.imshow('object detection', cv.resize(image_np, (800, 600)))
            print(image_tensor)
            print(boxes)
            print(scores)
            print(classes)
            print(num_detections)

            if cv.waitKey(25) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break