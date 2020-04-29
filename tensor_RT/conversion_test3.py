"""build_engine.py
This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.
Input : ssd_mobilenet_v[1|2]_[coco|egohands].pb
Output: TRT_ssd_mobilenet_v[1|2]_[coco|egohands].bin
"""


import os
import ctypes
import argparse

import uff
import tensorrt as trt
import graphsurgeon as gs


DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'libflattenconcat.so'))
MODEL_SPECS = {
    # 'ssd_mobilenet_v2_coco': {
    #     'input_pb':   os.path.abspath(os.path.join(
    #                       DIR_NAME, 'ssd_mobilenet_v2_coco.pb')),
    #     'tmp_uff':    os.path.abspath(os.path.join(
    #                       DIR_NAME, 'tmp_v2_coco.uff')),
    #     'output_bin': os.path.abspath(os.path.join(
    #                       DIR_NAME, 'TRT_ssd_mobilenet_v2_coco.bin')),
    #     'num_classes': 91,
    #     'min_size': 0.2,
    #     'max_size': 0.95,
    #     'input_order': [1, 0, 2],  # order of loc_data, conf_data, priorbox_data
    # },
    'ssd_mobilenet_v2_coco': {
        'input_pb':   '../models/mobilenet/optomized_model.pb',
        'tmp_uff':    'tmp_mobilenet_v2_coco.uff',
        'output_bin': 'TRT_ssd_mobilenet_v2_coco.bin',
        'num_classes': 3,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [1, 0, 2],  # order of loc_data, conf_data, priorbox_data
    },
}
INPUT_DIMS = (3, 300, 300)
DEBUG_UFF = False


def add_plugin(graph, model, spec):
    """add_plugin
    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """
    numClasses = spec['num_classes']
    minSize = spec['min_size']
    maxSize = spec['max_size']
    inputOrder = spec['input_order']

    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=(1,) + INPUT_DIMS
    )

    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=minSize,  # was 0.2
        maxSize=maxSize,  # was 0.95
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=numClasses,  # was 91
        inputOrder=inputOrder,
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    if trt.__version__[0] >= '7':
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
    else:
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT"
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT"
        )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
       