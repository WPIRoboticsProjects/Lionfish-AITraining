import json
import tensorflow as tf
import os
import random
import shutil
top_dir = []
directory = 'supervisely/Lionfish ID'
new_dir = 'data'

print(os.listdir(directory))
directory_list = os.listdir(directory)

# 20%
test_images = []

# 60%
train_images = []

# 20%
validate_images = []

for folder in directory_list:
    if 'meta.json' in folder:
        os.remove('{}/{}'.format(directory, folder))
    else:
        sub_dir = os.listdir('{}/{}'.format(directory, folder))
        print(sub_dir)
        for sub_folder in sub_dir:
            sub_sub_dir = os.listdir('{}/{}/{}'.format(directory, folder, sub_folder))
            print(sub_sub_dir)
            if sub_folder == 'img':
                sub_sub_sub_dir = os.listdir('{}/{}/{}'.format(directory, folder, sub_folder))

                for file in sub_sub_sub_dir:
                    sample = random.random()
                    if sample < .2:
                        test_images.append(('{}/{}/{}/{}'.format(directory, folder, sub_folder, file), file))
                    elif .2 <= sample < .4:
                        validate_images.append(('{}/{}/{}/{}'.format(directory, folder, sub_folder, file), file))
                    else:
                        train_images.append(('{}/{}/{}/{}'.format(directory, folder, sub_folder, file), file))
print(len(test_images), test_images)
print(len(validate_images), validate_images)
print(len(train_images), train_images)

images = 'data/images'
test = os.path.join(images, 'test')
train = os.path.join(images, 'train')
validate = os.path.join(images, 'validate')
for image in train_images:
    print(os.path.join(train, image[1]))
    shutil.move(image[0], os.path.join(train, image[1]))
    shutil.move(image[0].replace('img', 'ann') + '.json', os.path.join(train, image[1] + '.json'))
for image in test_images:
    print(os.path.join(train, image[1]))
    shutil.move(image[0], os.path.join(test, image[1]))
    shutil.move(image[0].replace('img', 'ann') + '.json', os.path.join(test, image[1] + '.json'))
for image in validate_images:
    print(os.path.join(train, image[1]))
    shutil.move(image[0], os.path.join(validate, image[1]))
    shutil.move(image[0].replace('img', 'ann') + '.json', os.path.join(validate, image[1] + '.json'))
