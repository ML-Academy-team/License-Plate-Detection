"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import pandas as pd
import tensorflow as tf
import argparse

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-t",
                    "--type_data",
                    help="type of input data files train/valid.",
                    default='train',
                    type=str)

parser.add_argument("-c",
                    "--csv_path",
                    help="Path of input .csv file.",
                    default=None,
                    type=str)


args = parser.parse_args()

def create_tf_example(row, path):
    filename = row['img_id']
    if(os.path.exists(os.path.join(path, '{}'.format(filename))) == False):
        return 0

    img_file = os.path.join(path, '{}'.format(filename))   

    with tf.io.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    image_format = b'jpg'
    filename = filename.encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    xmins.append(row['xmin'] / width)
    xmaxs.append(row['xmax'] / width)
    ymins.append(row['ymin'] / height)
    ymaxs.append(row['ymax'] / height)
    classes_text.append('licence_plate'.encode('utf8'))
    classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == '__main__':
    succuss_conv = 0
    if (args.type_data == 'train'):
        output_path = '/content/dataset/train.record'
        path = '/content/dataset/train_data/'
    elif (args.type_data == 'valid'):
        output_path = '/content/dataset/valid.record'
        path = '/content/dataset/val_data/'
    else:
        print('enter a valid type of data')
        exit
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(path)
    examples = pd.read_csv(args.csv_path)

    for index, row in examples.iterrows():
        tmp = create_tf_example(row, path)
        if(tmp == 0):
            continue
        writer.write(tmp.SerializeToString())
        succuss_conv +=1
  
    writer.close()

    print('Successfully created the TFRecords: {} with {} succuss record conversion'.format(output_path, succuss_conv))
