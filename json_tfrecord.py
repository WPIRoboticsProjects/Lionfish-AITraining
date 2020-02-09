import json
import tensorflow as tf
import os
top_dir = []
data = 'data'

for x in os.walk(data):
    top_dir = x[1]
    break
print(top_dir)

for folder in top_dir:
    for sub_folder in os.walk(data + '/' + folder):
        print(sub_folder)

    print('here')
