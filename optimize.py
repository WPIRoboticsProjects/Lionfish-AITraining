import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
import os
import numpy as np

from tensorflow_core.python.data import ops


def get_graph_def_from_saved_model(saved_model_dir):
  with tf.Session() as session:
    meta_graph_def = tf.saved_model.loader.load(
    session,
    tags=[tag_constants.SERVING],
    export_dir=saved_model_dir
  )
  return meta_graph_def.graph_def

def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format(
      [node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format(
      [node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format(
      [node.name for node in graph_def.node if (
          'predictions' in node.name or 'softmax' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format(
      [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format(
      len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Identity'])))
  print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))

def get_size(model_dir, model_file='saved_model.pb'):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if os.path.exists(
      os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(
        model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
  print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))

def get_graph_def_from_file(graph_filepath):
  with tf.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms, output_node):
  input_names = ['image_tensor'] #[]
  # output_names = [output_node]
  output_names = ['detection_boxes', 'detection_scores','detection_multiclass_scores', 'detection_classes', 'num_detections']
  if graph_filename is None:
    graph_def = get_graph_def_from_saved_model(model_dir)
  else:
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name='optimized_model.pb')
  print('Graph optimized!')


def load_graph(model_filepath):
    '''
    Lode trained model.
    '''
    print('Loading model...')
    graph = tf.Graph()

    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)

    with graph.as_default():
        # Define input tensor
        # input = tf.placeholder(np.float32, shape=[None, 32, 32, 3], name='input')
        # dropout_rate = tf.placeholder(tf.float32, shape=[], name='dropout_rate')
        tf.import_graph_def(graph_def) #, {'input': input, 'dropout_rate': dropout_rate})

    graph.finalize()

    print('Model loading complete!')

    # Get layer names
    layers = [op.name for op in graph.get_operations()]
    for layer in layers:
        print(layer)

    """
    # Check out the weights of the nodes
    weight_nodes = [n for n in graph_def.node if n.op == 'Const']
    for n in weight_nodes:
        print("Name of the node - %s" % n.name)
        # print("Value - " )
        # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    """

    # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
    # self.sess = tf.InteractiveSession(graph = self.graph)
    sess = tf.Session(graph=graph)

if __name__ == "__main__":
    # model = get_graph_def_from_saved_model('../../../Downloads/mobilenetModelTest/mobilenet_output_inference_graph.pb/frozen_inference_graph.pb')
    # describe_graph(model)
    # get_size('../../../Downloads/mobilenetModelTest/mobilenet_output_inference_graph.pb/frozen_inference_graph.pb')
    # load_graph('../../../Downloads/mobilenetModelTest/mobilenet_output_inference_graph.pb/frozen_inference_graph.pb')

    transforms = ['remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms']

    # optimize_graph('../../../Downloads/mobilenetModelTest/mobilenet_output_inference_graph.pb', 'frozen_inference_graph.pb', transforms, ['import/detection_boxes', 'import/detection_scores','import/detection_multiclass_scores', 'import/detection_classes', 'import/num_detections'])
    describe_graph(get_graph_def_from_file('../../../Downloads/mobilenetModelTest/mobilenet_output_inference_graph.pb/optimized_model.pb'))
