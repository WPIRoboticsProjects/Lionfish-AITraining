import numpy as np
import tensorflow as tf
import cv2 as cv
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
# import os
# os.chdir("/home/nicholas/Downloads/models/research/object_detection")
from object_detection.utils import visualization_utils as vis_util # here
from object_detection.utils import label_map_util # here
# from utils import visualization_utils as vis_util

# PATH_TO_MODEL = 'models/mobilenet/optimized_model.pb'
# PATH_TO_MODEL = 'models/inception/inception_frozen.pb'
PATH_TO_MODEL = 'models/resnet/resnet_frozen.pb'
PATH_TO_LABELS = 'models/mobilenet/data-inception-lionfish_lionfish_label_map.pbtxt'
NUM_CLASSES = 3

# some code from: https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b

def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_score={}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_score.keys():
                model_score[score]=[img_id]
            else:
                model_score[score].append(img_id)
    return model_score


def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if x_topleft_gt > x_bottomright_gt:
        print("\nWarning!!")
        print("Ground Truth Bounding Box is not correct", x_topleft_gt, x_bottomright_gt, y_topleft_gt, y_bottomright_gt)
        print("")
        temp = x_topleft_gt
        x_topleft_gt = x_bottomright_gt
        x_bottomright_gt = temp

    if y_topleft_gt > y_bottomright_gt:
        print("\nWarning!!")
        print("Ground Truth Bounding Box is not correct", x_topleft_gt, x_bottomright_gt, y_topleft_gt, y_bottomright_gt)
        print("")
        temp = y_topleft_gt
        y_topleft_gt = y_bottomright_gt
        y_bottomright_gt = temp


    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

        return 0.0
    if (y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

        return 0.0
    if (x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

        return 0.0
    if (y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area / union_area

def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive +=res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def get_avg_precision_at_iou(gt_boxes, pred_bb, iou_thr=0.5):
    model_scores = get_model_scores(pred_bb)
    sorted_model_scores = sorted(model_scores.keys())
    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_bb.keys():
        arg_sort = np.argsort(pred_bb[img_id]['scores'])
        pred_bb[img_id]['scores'] = np.array(pred_bb[img_id]['scores'])[arg_sort].tolist()
        pred_bb[img_id]['boxes'] = np.array(pred_bb[img_id]['boxes'])[arg_sort].tolist()


    pred_boxes_pruned = deepcopy(pred_bb)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        print("Model score : ", model_score_thr)
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
    # indent start
        for img_id in img_ids:

            gt_boxes_img = gt_boxes[img_id] # ['boxes'] # change here by adding boxes
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break
                    # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]
            # Recalculate image results for this image
            # print(img_id)
            # print(gt_boxes_img)
            # print(pred_boxes_pruned[img_id]['boxes'])
            img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr=0.5)
            # calculate precision and recall
        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)
    # indent end
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls > recall_level).flatten()
            prec = max(precisions[args])
            print(recalls, "Recall")
            print(recall_level, "Recall Level")
            print(args, "Args")
            print(prec, "precision")
        except ValueError:
            print("value error")
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}

def readData():
    dataset = tf.data.TFRecordDataset(filenames=['test.record'])

    image_array = []
    id_array = []
    coord_array = []
    feature_array = []

    data_iterator = iter(dataset)

    zero = 0
    single = 0
    multiple = 0
    total = 0

    while True:
        try:
            raw_example = next(data_iterator)
            parsed = tf.train.Example.FromString(raw_example.numpy())
            # print(parsed)

            raw_img = parsed.features.feature['image/encoded'].bytes_list.value[0]
            format = parsed.features.feature['image/format'].bytes_list.value[0].decode("utf-8")
            if format == "jpg":
                image = tf.image.decode_jpeg(raw_img)
            elif format == "png":
                image = tf.image.decode_png(raw_img)
            elif format == "bmp":
                image = tf.image.decode_bmp(raw_img)
            height = parsed.features.feature['image/height'].int64_list.value[0]
            width = parsed.features.feature['image/width'].int64_list.value[0]
            id = parsed.features.feature['image/source_id'].bytes_list.value[0].decode("utf-8").split("/")[-1]
            label = parsed.features.feature['image/object/class/label'].int64_list.value
            feature_tmp = parsed.features.feature['image/object/class/text'].bytes_list.value
            feature = []
            for feat in feature_tmp:
                feature.append(feat.decode("utf-8"))
            xmax = parsed.features.feature['image/object/bbox/xmax'].float_list.value
            xmin = parsed.features.feature['image/object/bbox/xmin'].float_list.value
            ymax = parsed.features.feature['image/object/bbox/ymax'].float_list.value
            ymin = parsed.features.feature['image/object/bbox/ymin'].float_list.value
            # print("id: ", id)
            # print("label: ", label)
            # print("feature: ", feature)
            # print("xmax: ", xmax)
            # print("ymax: ", ymax)
            # print("xmin: ", xmin)
            # print("ymin: ", ymin)

            temp_coord_array = []
            for i in range(0, len(feature)):
                temp_coord_array.append([xmin[i]*300.0, ymin[i]*300.0, xmax[i]*300.0, ymax[i]*300.0])

            coord_array.append(temp_coord_array)
            id_array.append(str(id))
            feature_array.append(feature)

            # print("width: ", width)
            # print("height: ", height)

            image_array.append(image)

            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
            if len(feature) == 0:
                zero += 1
            elif len(feature) == 1:
                single += 1
            else:
                multiple += 1
            total += 1
            # print(image)
        except StopIteration:
            break

        print("")
        print("zero: ", zero)
        print("single: ", single)
        print("multiple: ", multiple)
        print("total: ", total)

    return image_array, id_array, coord_array, feature_array

def lookup_labels(score_array, class_array, num):
    # print(score_array)
    # print(class_array)
    # print(num)
    # print("")
    return_score_array = []
    return_class_array = []
    for i in range(0, num):
        if class_array[i] == 1:
            return_score_array.append(score_array[i])
            return_class_array.append("Lionfish")
        elif class_array[i] == 2:
            return_score_array.append(score_array[i])
            return_class_array.append("Diver")
        else:
            print("If reached, background data is in here")
            return_score_array.append(score_array[i])
            return_class_array.append("Background")
    return return_score_array, return_class_array

def run_detection(detection_graph, image_array):

    actual_detections = []
    actual_scores = []
    actual_labels = []

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS) # here
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories) # to here

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # Detection
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
            i = 0
            while i < len(image_array):

                # # Read frame from camera
                # ret, img = cap.read()
                # cv.imwrite((str(i) + ".png"), cv.cvtColor(np.array(image_array[i]), cv.COLOR_BGR2RGB))
                img = cv.cvtColor(np.array(image_array[i]), cv.COLOR_BGR2RGB)
                img = cv.resize(img, (300, 300))
                image_np = np.asarray(img) # .astype('uint8')

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
                # Visualization of the results of a detection. # here
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4) # to here

                # Display output
                # cv.imshow('object detection', cv.resize(image_np, (800, 600)))
                cv.imwrite(("out_images/" + str(i) + ".png"), cv.resize(image_np, (800, 600)))
                # print(image_tensor)
                # print(np.squeeze(boxes))
                temp_detection = []
                for box in np.squeeze(boxes):
                    if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
                        # nothing found
                        pass
                    else:
                        new_detection = [box[0]*300.0, box[1]*300.0, box[2]*300.0, box[3]*300.0]
                        temp_detection.append(new_detection)
                actual_detections.append(temp_detection)

                comp_score, comp_class = lookup_labels(np.squeeze(scores), np.squeeze(classes), len(temp_detection))
                actual_scores.append(comp_score)
                actual_labels.append(comp_class)

                # print(np.squeeze(scores))
                # print(classes)
                # print(num_detections)
                print("Curr pred # " + str(i))
                i += 1
    # print("\nActual Detections: ")
    # print(actual_detections)
    # print("\nActual Scores: ")
    # print(actual_scores)
    # print("\nActual labels: ")
    # print(actual_labels)
    return actual_detections, actual_scores, actual_labels

def clean_data(ground_id, ground_coord, ground_feature, det_coord, det_scores, det_feature):
    ground_boxes = {}
    pred_boxes = {}

    for i in range(0, len(ground_id)):
        # ground_boxes[ground_id[i]] = {}
        # ground_boxes[ground_id[i]]["boxes"] = ground_coord[i]
        # ground_boxes[ground_id[i]]["features"] = ground_feature[i]
        ground_boxes[ground_id[i]] = ground_coord[i]
        pred_boxes[ground_id[i]] = {}
        pred_boxes[ground_id[i]]["boxes"] = det_coord[i]
        pred_boxes[ground_id[i]]["features"] = det_feature[i]
        pred_boxes[ground_id[i]]["scores"] = det_scores[i]
    print("\nGround: ")
    print(ground_boxes)
    print("\nPredicted: ")
    print(pred_boxes)
    return ground_boxes, pred_boxes


if __name__== "__main__":
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    image_array, id_array, coord_array, feature_array = readData()
    # print("\nIds: ")
    # print(id_array)
    # print("\nCoords: ")
    # print(coord_array)
    # print("\nFeatures: ")
    # print(feature_array)

    detections, scores, labels = run_detection(detection_graph, image_array)

    ground, predicted = clean_data(id_array, coord_array, feature_array, detections, scores, labels)

    print("\nGround ")
    print(ground)
    print("\nPredicted ")
    print(predicted)

    precision_data = get_avg_precision_at_iou(ground, predicted)

    print("\nPrecision data: ")
    print(precision_data)

