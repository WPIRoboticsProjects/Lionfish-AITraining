import os
import glob
import pandas as pd
import json

def json_to_csv(path):
    json_list = []
    for json_file in glob.glob(path + '/*.json'):
        with open(json_file, 'r') as file:
            data = json.load(file)
            size = data['size']

        for member in data['objects']:
            box = member['points']
            value = (json_file[0:len(json_file) - 5],
                     int(size['width']),
                     int(size['height']),
                     member['classTitle'],
                     int(box['exterior'][0][0]),
                     int(box['exterior'][0][1]),
                     int(box['exterior'][1][0]),
                     int(box['exterior'][1][1])
                     )
            json_list.append(value)
        print(json_list)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    json_df = pd.DataFrame(json_list, columns=column_name)
    return json_df


def main():
    print(os.getcwd())
    # Todo add validate folder
    for directory in ['train', 'validate', 'test']:
        image_path = os.path.join(os.getcwd(), 'data/images/{}'.format(directory))
        json_df = json_to_csv(image_path)
        json_df.to_csv('data/{}.csv'.format(directory), index=None)
        print('Successfully converted json to csv.')


main()