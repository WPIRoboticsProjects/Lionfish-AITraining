from tensorflow.python.compiler.tensorrt import trt_convert as trt
# Convert a saved model
converter = trt.TrtGraphConverterV2(input_saved_model_dir='../models/mobilenet_output_inference_graph/saved_model')
graph_def = converter.convert()
converter.save('mobilenetTRT/')
