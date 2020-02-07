import json
from sys import argv
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def parseFiles(directory):
    imageFolders = []
    imageDirectory = ""
    number = 0

    for file in os.listdir(directory):
        print(file)
        if file.split(".")[-1] == "json":
            # meta data
            pass
        else:
            # image frames
            print("image frames")
            imageDirectory = os.path.join(directory, file)
            imageFolders.append(imageDirectory)
            print(imageDirectory)

    print("")

    writer = tf.compat.v1.python_io.TFRecordWriter("output/test1.tfrecord")

    for folder in imageFolders:
        print(folder)
        for file in os.listdir(os.path.join(folder, "img")): # os.listdir(os.path.join(imageDirectory, "img")):
            print(file)
            try:
                # print(imageDirectory + "\\img\\" + file)
                # image = open(os.path.join(imageDirectory, "img", file), "r")
                # print(imageDirectory + "\\ann\\" + file + ".json")
                # annotation = open(os.path.join(imageDirectory, "ann", file) + ".json", "r")

                # print(folder + "\\img\\" + file)
                # image = open(os.path.join(folder, "img", file), "r")
                # print(imageDirectory + "\\ann\\" + file + ".json")
                annotation = open(os.path.join(folder, "ann", file) + ".json", "r")

                annotationJson = json.load(annotation)
                print(annotationJson)

                width = annotationJson["size"]["width"]
                height = annotationJson["size"]["height"]

                tags = []
                if len(annotationJson["tags"]) > 0:
                    # there are tags
                    for obj in annotationJson["tags"]:
                        tags.append(obj["name"])


                objects = []

                x_mins = []
                x_maxs = []
                y_mins = []
                y_maxs = []

                classes_text = []
                classes_int = []

                if len(annotationJson["objects"]) > 0:
                    # there are objects
                    for obj in annotationJson["objects"]:
                        points = obj["points"]["exterior"]
                        x_mins.append(points[0][0])
                        x_maxs.append(points[1][0])
                        y_mins.append(points[1][1])
                        y_maxs.append(points[0][1])

                        classes_text.append(obj["classTitle"])
                        if obj["classTitle"] == "Diver":
                            classes_int.append(1)
                        else:
                            classes_int.append(0)

                        # objects.append((obj["classTitle"], (points[0][0], points[0][1]), (points[1][0], points[1][1])))

                print("H: " + str(height))
                print("W: " + str(width))
                print("Tags: " + str(tags))
                # print("Objects: " + str(objects))

                imagePath = os.path.join(folder, "img", file)
                tfrecordWrite(writer, imagePath, number, x_mins, x_maxs, y_mins, y_maxs, classes_text, classes_int)

                number += 1

                # image.close()
                annotation.close()

                print("")
            except Exception as e:
                print(e)
                print("Failed on: " + file)
    writer.close()

def tfrecordWrite(in_writer, imagePath, number, x_mins, x_maxs, y_mins, y_maxs, classes_text, classes_int):

    writer = in_writer

    with tf.compat.v1.gfile.GFile(imagePath, 'rb') as f:
        im_data = f.read()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/raw': _bytes_feature(tf.compat.as_bytes(im_data)),
        'image/object/box/xmins': int64_list_feature(x_mins),
        'image/object/box/xmaxs': int64_list_feature(x_maxs),
        'image/object/box/ymins': int64_list_feature(y_mins),
        'image/object/box/ymaxs': int64_list_feature(y_maxs),
        # 'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes_int)
    }))

    writer.write(example.SerializeToString())

def readtfrecord(filepath):

    dataset = tf.data.TFRecordDataset(filepath)

    for element in dataset.__iter__():
        parsed = tf.train.Example.FromString(element.numpy())
        print(parsed.features.feature['image/object/box/xmins'].int64_list.value)
        print(parsed.features.feature['image/object/box/xmaxs'].int64_list.value)
        print(parsed.features.feature['image/object/box/ymins'].int64_list.value)
        print(parsed.features.feature['image/object/box/ymaxs'].int64_list.value)
        print(parsed.features.feature['image/object/class/label'].int64_list.value)
        raw_img = parsed.features.feature['image/raw'].bytes_list.value[0]
        img = tf.image.decode_png(raw_img)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        print()





def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def main():
    if len(argv) != 2:
        print("incorrect number of arguments")
        print("python3 preprocess.py <directory>")
        return

    directory = argv[1]
    parseFiles(directory)

    readtfrecord("output/test1.tfrecord")


if __name__ == '__main__':
    main()
