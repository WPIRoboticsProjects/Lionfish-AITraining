from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import argparse
import os

def freeze_model(dir, pbtxt, ckpt, output):
    first = os.path.join(dir, pbtxt)
    second = os.path.join(dir, ckpt)
    out = os.path.join(dir, output)

    freeze_graph.freeze_graph(first, "", False,
                              second, "output_node",
                              "save/restore_all", "save/Const:0",
                              out, True, ""
                              )

    # freeze_graph.freeze_graph('tensorflowModel.pbtxt', "", False,
    #                       './tensorflowModel.ckpt', "output/softmax",
    #                        "save/restore_all", "save/Const:0",
    #                        'frozentensorflowModel.pb', True, ""
    #                      )

def optimize_model():
    pass

def main():
    print(tf.__version__)
    parser = argparse.ArgumentParser(description='Convert model to Pb')
    parser.add_argument('--dir', type=str, help='directory of the model to convert', required=True)
    parser.add_argument('--pbtxt', type=str, help='name of the pbtxt file', required=True)
    parser.add_argument('--ckpt', type=str, help='name of the ckpt file', required=True)
    parser.add_argument('--output', type=str, help='name of the output', required=True)
    args = parser.parse_args()
    print(args.dir)
    print(args.pbtxt)
    print(args.ckpt)
    print(args.output)
    freeze_model(args.dir, args.pbtxt, args.ckpt, args.output)


if __name__ == '__main__':
    main()