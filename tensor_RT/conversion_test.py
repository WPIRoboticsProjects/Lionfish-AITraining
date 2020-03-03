from tensorflow.python.compiler.tensorrt import trt_convert as trt
# Convert a saved model
converter = trt.TrtGraphConverter(input_saved_model_dir='../mobilenet_output_inference_graph')
graph_def = converter.convert()
converter.save('mobilenetTRT/')
