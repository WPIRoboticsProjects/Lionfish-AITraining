from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
import tensorflow as tf

def getTrtGraphDef():
    converter = trt.TrtGraphConverterV2(input_saved_model_dir='../models/mobilenet_output_inference_graph/saved_model')
    converter.convert()
    converter.build()
    converter.save('mobilenetTRTFull/')
    saved_model_loaded = tf.saved_model.load('mobilenetTRTFull/')
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func.graph.as_graph_def()


trt_graph = getTrtGraphDef()
for n in trt_graph.node:
  if n.op == "TRTEngineOp":
    print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
    with tf.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
      f.write(n.attr["serialized_segment"].s)
  else:
    print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))