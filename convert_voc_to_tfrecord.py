import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import PIL.Image
import io

# Paths
data_dir = '/home/object_detection/data/voc/VOCdevkit/VOC2012'
train_output_path = '/home/object_detection/data/voc/voc_train.record'
val_output_path = '/home/object_detection/data/voc/voc_val.record'

# VOC label map dictionary
label_map_dict = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

# Convert a VOC XML file to a tf.Example
def dict_to_tf_example(data, dataset_directory, label_map_dict):
    img_path = os.path.join(dataset_directory, 'JPEGImages', data['filename'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    if 'object' in data:
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin']) / width
            xmax = float(obj['bndbox']['xmax']) / width
            ymin = float(obj['bndbox']['ymin']) / height
            ymax = float(obj['bndbox']['ymax']) / height

            if xmin >= xmax or ymin >= ymax:
                continue

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Create TFRecord file
def create_tf_record(output_path, data_dir, label_map_dict, subset):
    writer = tf.io.TFRecordWriter(output_path)

    annotations_dir = os.path.join(data_dir, 'Annotations')
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(annotations_dir, xml_file)
        with tf.io.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        # Check if the current file is for training or validation
        if subset in data['filename']:
            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created the TFRecord file: {output_path}')

# Main function
def main():
    create_tf_record(train_output_path, data_dir, label_map_dict, subset='train') # Create training TFRecord file
    create_tf_record(val_output_path, data_dir, label_map_dict, subset='val') # Create validation TFRecord file

if __name__ == '__main__':
    main()
